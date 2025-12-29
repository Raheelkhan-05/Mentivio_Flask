# services/socratic_tutor.py
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from services.document_processor import DocumentProcessor
from database.messages import MongoMessageDB
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import logging
import time
import re

load_dotenv()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

RETRIEVE_TOP_K = 20
MAX_CONTEXT_CHUNKS = 15
MAX_HISTORY_MESSAGES = 10
CONTEXT_WINDOW_CHARS = 10000

class SocraticTutor:
    """Production Socratic tutor with natural conversation."""
    
    def __init__(self):
        logger.info("[INIT] Socratic Tutor")
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_API_NAME"),
            model_name=os.getenv("AZURE_OPENAI_MODEL", "gpt-4o"),
            temperature=0.7
        )
        self.doc_processor = DocumentProcessor()
        self.db = MongoMessageDB()
        # Track state per chat
        self.chat_states = {}

    def _get_state(self, chat_id: str) -> Dict:
        """Get or create chat state."""
        if chat_id not in self.chat_states:
            self.chat_states[chat_id] = {
                "current_topic": None,
                "probes_asked": 0,  # Track how many times we probed without answer
                "got_good_answer": False
            }
        return self.chat_states[chat_id]

    def _build_context(self, chunks: List[Dict]) -> str:
        if not chunks:
            return ""
        sorted_chunks = sorted(chunks, key=lambda x: x.get("similarity", 0), reverse=True)
        context_parts = []
        total_chars = 0
        for chunk in sorted_chunks[:MAX_CONTEXT_CHUNKS]:
            content = chunk.get("content", "").strip()
            if content and total_chars < CONTEXT_WINDOW_CHARS:
                context_parts.append(content)
                total_chars += len(content)
        return "\n\n".join(context_parts)

    def _get_history(self, chat_id: str) -> List:
        try:
            msgs = self.db.get_last_messages(chat_id, MAX_HISTORY_MESSAGES)
            return [
                HumanMessage(content=m["content"]) if m["role"] == "user" 
                else AIMessage(content=m["content"])
                for m in msgs
            ]
        except:
            return []

    def _extract_topic(self, text: str) -> str:
        """Extract topic from question."""
        t = text.lower().strip()
        # Remove question words
        t = re.sub(r'^(what is|what are|whats|what\'s|explain|tell me about|describe|how does|why does)\s+', '', t)
        t = re.sub(r'\?+$', '', t).strip()
        
        if t and len(t) > 2:
            return ' '.join(t.split()[:4])  # Max 4 words
        return text.strip()[:50]

    def _is_question_about_topic(self, text: str) -> bool:
        """Check if text is a new question."""
        lower = text.lower().strip()
        # Starts with question word or ends with ?
        return bool(re.match(r'^(what|how|why|when|where|who|explain|tell|describe)', lower)) or text.endswith('?')

    def _is_confusion(self, text: str) -> bool:
        """Check if student is confused."""
        lower = text.lower().strip()
        return any(phrase in lower for phrase in [
            "don't know", "dont know", "idk", "no idea", 
            "not sure", "confused", "don't understand"
        ]) or lower in ["no", "nope", "nah"]

    def _is_good_answer(self, text: str, topic: str) -> bool:
        """Check if student gave a substantive answer."""
        lower = text.lower().strip()
        words = lower.split()
        
        # Must be at least 5 words
        if len(words) < 5:
            return False
        
        # Check if mentions topic
        topic_words = set(topic.lower().split())
        text_words = set(words)
        has_topic = bool(topic_words & text_words)
        
        # Check for explanation patterns
        has_explanation = any(pattern in lower for pattern in [
            "is a", "is used", "used for", "used to", "is when",
            "helps", "allows", "creates", "runs", "works", 
            "means", "refers to", "involves"
        ])
        
        return has_topic and has_explanation

    def generate_questions(
        self,
        student_question: str,
        user_id: str,
        chat_id: str,
        material_id: Optional[str],
        use_all_materials: bool
    ) -> Dict:
        """Main generation method."""
        start = time.time()
        
        try:
            if not student_question or not chat_id:
                return {"answer": "What would you like to learn about?", "sources": [], "mode": "error"}
            
            if not user_id:
                use_all_materials = False
                material_id = None
            
            # Save user message
            if user_id:
                self.db.add_message(chat_id, user_id, "user", student_question)
            
            # Get state
            state = self._get_state(chat_id)
            
            # Determine what type of input this is
            is_new_question = self._is_question_about_topic(student_question)
            is_confused = self._is_confusion(student_question)
            
            topic = self._extract_topic(student_question)
            
            # Decision logic
            if is_new_question:
                # New topic - reset state
                state["current_topic"] = topic
                state["probes_asked"] = 0
                state["got_good_answer"] = False
                mode = "initial_probe"
                logger.info(f"[NEW TOPIC] {topic}")
            
            elif is_confused:
                # Student doesn't know
                state["probes_asked"] += 1
                logger.info(f"[CONFUSED] Probe count: {state['probes_asked']}")
                
                if state["probes_asked"] >= 2:
                    # Give full answer after 2 failed attempts
                    mode = "full_explanation"
                    state["got_good_answer"] = True
                else:
                    # Simplify question
                    mode = "simpler_probe"
            
            elif self._is_good_answer(student_question, state.get("current_topic", "")):
                # Student gave good answer!
                state["got_good_answer"] = True
                state["probes_asked"] = 0
                mode = "celebrate"
                logger.info(f"[GOOD ANSWER] Topic: {state['current_topic']}")
            
            else:
                # Some other response
                if state["probes_asked"] >= 2:
                    mode = "full_explanation"
                    state["got_good_answer"] = True
                else:
                    state["probes_asked"] += 1
                    mode = "follow_up"
            
            # Get context
            chunks = []
            if user_id:
                try:
                    chunks = self.doc_processor.get_relevant_chunks(
                        query=student_question,
                        user_id=user_id,
                        material_id=material_id,
                        use_all_materials=use_all_materials,
                        top_k=RETRIEVE_TOP_K
                    ) or []
                except Exception as e:
                    logger.warning(f"[CHUNKS] Error: {e}")
            
            context = self._build_context(chunks)
            history = self._get_history(chat_id)
            
            # Generate response
            answer = self._generate(
                mode=mode,
                student_input=student_question,
                topic=state.get("current_topic", topic),
                context=context,
                history=history,
                probe_count=state["probes_asked"]
            )
            
            # Save bot message
            if user_id:
                self.db.add_message(chat_id, user_id, "assistant", answer)
            
            # Sources
            sources = []
            if chunks:
                for c in chunks[:3]:
                    sources.append({
                        "content": c.get("content", "")[:200],
                        "similarity": round(c.get("similarity", 0), 3),
                        "metadata": c.get("metadata", {})
                    })
            
            elapsed = time.time() - start
            logger.info(f"[DONE] {mode} in {elapsed:.2f}s")
            
            return {"answer": answer, "sources": sources, "mode": mode}
            
        except Exception as e:
            logger.exception(f"[ERROR] {e}")
            return {"answer": "Let me know what you'd like to explore!", "sources": [], "mode": "error"}

    def _generate(
        self,
        mode: str,
        student_input: str,
        topic: str,
        context: str,
        history: List,
        probe_count: int
    ) -> str:
        """Generate response based on mode."""
        
        # Get last few exchanges for context
        recent = ""
        if history:
            last_4 = history[-4:]
            lines = []
            for msg in last_4:
                if isinstance(msg, HumanMessage):
                    lines.append(f"Student: {msg.content}")
                elif isinstance(msg, AIMessage):
                    lines.append(f"You: {msg.content}")
            recent = "\n".join(lines)
        
        if mode == "initial_probe":
            # First question - assume they know something
            prompt = f"""You're a casual tutor. Student asked: "{student_input}"

Topic: {topic}

Assume they've heard about this before. Ask what they know in a natural, friendly way.

Examples (pick a style):
- "What do you know about {topic} already?"
- "Have you worked with {topic} before?"
- "What have you seen {topic} used for?"

Just 1 short question. Sound like a friend, not a teacher. Don't say "Great question!" or "Interesting!"."""

        elif mode == "simpler_probe":
            # They said "I don't know" once - ask simpler
            prompt = f"""Topic: {topic}
Student said: "{student_input}" (they don't know)
Probe count: {probe_count}/2

Recent chat:
{recent}

Give an encouraging response with a SIMPLER question. Make it easier.

Examples:
- "That's okay! Have you used any apps that might use {topic}?"
- "No worries! Think about programs you use - any connection to {topic}?"

Keep it to 2 sentences max. Be encouraging and casual."""

        elif mode == "follow_up":
            # They gave some response but not complete
            prompt = f"""Topic: {topic}
Student said: "{student_input}"
Probe count: {probe_count}/2

Recent chat:
{recent}

They gave an answer but need to expand. Ask a natural follow-up.

Examples:
- "Interesting! What else do you know about {topic}?"
- "Got it. Where have you seen {topic} used?"

1-2 sentences. Keep it conversational."""

        elif mode == "full_explanation":
            # They don't know after 2 tries - explain fully
            prompt = f"""Topic: {topic}
Student said: "{student_input}" 
They've said "don't know" {probe_count} times. Time to explain clearly.

Recent chat:
{recent}

Context material:
{context}

Write a clear, friendly explanation:
1. Start: "No problem! Let me explain."
2. Explain what {topic} is (2-3 simple sentences)
3. Give a concrete example
4. End: "Make sense?"

Be conversational, not textbook-like. Use the context for accuracy."""

        elif mode == "celebrate":
            # They got it right! Celebrate and add knowledge
            prompt = f"""IMPORTANT: Student correctly explained {topic}!

Their answer: "{student_input}"

Recent chat:
{recent}

Context material:
{context}

Your response:
1. Celebrate enthusiastically (1 sentence): "Exactly!" or "Spot on!" or "Yes, that's right!"
2. Acknowledge their answer briefly
3. Add 2-3 NEW interesting facts from the context they don't know
4. Ask if they want to know more: "Want to dive deeper?" or "Curious about anything else?"

Be natural and enthusiastic. Don't ask them to explain more - they already did!"""

        else:
            prompt = f"""Topic: {topic}
Student: "{student_input}"

Respond naturally as a friendly tutor. Keep it conversational."""
        
        # Build messages
        messages = [
            SystemMessage(content="You're a friendly, casual tutor. Keep responses natural and conversational, not robotic or templated."),
            HumanMessage(content=prompt)
        ]
        
        # Generate
        response = self.llm.invoke(messages)
        return response.content.strip()