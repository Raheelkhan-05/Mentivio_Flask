# services/qa_service.py
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from services.document_processor import DocumentProcessor
from database.messages import MongoMessageDB
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple
import logging
import time
import re
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

load_dotenv()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Tunable constants
RETRIEVE_TOP_K = 30
MAX_CONTEXT_CHUNKS = 20
MAX_HISTORY_MESSAGES = 20
CONTEXT_WINDOW_CHARS = 12000
SEMANTIC_THRESHOLD = 0.3

class ConversationMemory:
    """Track user-stated facts with enhanced parsing."""
    def __init__(self):
        self.user_facts = {}
        self.last_topic = {}
    
    def set_fact(self, chat_id: str, key: str, value: str):
        if chat_id not in self.user_facts:
            self.user_facts[chat_id] = {}
        self.user_facts[chat_id][key] = value
        logger.info(f"[MEMORY] Set fact for chat {chat_id}: {key} = {value}")
    
    def get_fact(self, chat_id: str, key: str) -> Optional[str]:
        value = self.user_facts.get(chat_id, {}).get(key)
        logger.info(f"[MEMORY] Get fact for chat {chat_id}: {key} = {value}")
        return value
    
    def get_all_facts(self, chat_id: str) -> Dict:
        facts = self.user_facts.get(chat_id, {})
        logger.info(f"[MEMORY] All facts for chat {chat_id}: {facts}")
        return facts
    
    def set_last_topic(self, chat_id: str, topic: str):
        self.last_topic[chat_id] = topic
        logger.info(f"[MEMORY] Set last topic for chat {chat_id}: {topic}")
    
    def get_last_topic(self, chat_id: str) -> Optional[str]:
        return self.last_topic.get(chat_id)
    
    def clear_facts(self, chat_id: str):
        if chat_id in self.user_facts:
            del self.user_facts[chat_id]
        if chat_id in self.last_topic:
            del self.last_topic[chat_id]
        logger.info(f"[MEMORY] Cleared all facts for chat {chat_id}")

class SemanticAttention:
    """Memory-optimized semantic attention using smaller model."""
    def __init__(self, model_name='paraphrase-MiniLM-L3-v2'):
        logger.info(f"[ATTENTION] Initializing SemanticAttention with model: {model_name}")
        # Use smaller model and optimize memory
        self.model = SentenceTransformer(model_name)
        self.device = 'cpu'  # Force CPU to save memory
        self.model.to(self.device)
        
        # Set model to eval mode and disable gradient computation
        self.model.eval()
        torch.set_grad_enabled(False)
        
        logger.info(f"[ATTENTION] Model loaded on device: {self.device}")
    
    def compute_attention_scores(self, query: str, texts: List[str]) -> np.ndarray:
        """Compute attention scores with memory optimization."""
        if not texts:
            return np.array([])
        
        logger.info(f"[ATTENTION] Computing attention for query over {len(texts)} texts")
        
        with torch.no_grad():  # Disable gradients to save memory
            # Encode query and texts
            query_embedding = self.model.encode(
                query, 
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=1  # Process one at a time to save memory
            )
            text_embeddings = self.model.encode(
                texts, 
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=8  # Smaller batch size
            )
            
            # Compute cosine similarity
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0), 
                text_embeddings
            ).cpu().numpy()
        
        logger.info(f"[ATTENTION] Scores - min: {similarities.min():.3f}, max: {similarities.max():.3f}")
        
        return similarities
    
    def select_relevant_context(self, query: str, history: List, max_msgs: int = 8) -> List:
        """Select most relevant messages using semantic attention."""
        if not history or len(history) <= max_msgs:
            logger.info(f"[ATTENTION] History length {len(history)} <= {max_msgs}, returning all")
            return history
        
        # Extract text content from messages
        msg_texts = []
        for msg in history:
            if hasattr(msg, 'content'):
                msg_texts.append(msg.content)
            else:
                msg_texts.append(str(msg))
        
        # Compute attention scores
        attention_scores = self.compute_attention_scores(query, msg_texts)
        
        # Always include last 5 messages (recency bias)
        relevant_indices = set(range(max(0, len(history) - 5), len(history)))
        logger.info(f"[ATTENTION] Including last 5 messages by default")
        
        # Add most relevant messages based on attention scores
        ranked_indices = np.argsort(attention_scores)[::-1]
        for idx in ranked_indices:
            if attention_scores[idx] > SEMANTIC_THRESHOLD or len(relevant_indices) < 4:
                relevant_indices.add(idx)
                logger.info(f"[ATTENTION] Adding message {idx} with score {attention_scores[idx]:.3f}")
            if len(relevant_indices) >= max_msgs:
                break
        
        selected = sorted(relevant_indices)
        logger.info(f"[ATTENTION] Selected {len(selected)} messages")
        return [history[i] for i in selected]

class QAService:
    def __init__(self):
        logger.info("[QA_SERVICE] Initializing QAService")
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_API_NAME"),
            model_name=os.getenv("AZURE_OPENAI_MODEL", "gpt-4o"),
            temperature=float(os.getenv("QA_TEMPERATURE", 0.3))
        )
        self.doc_processor = DocumentProcessor()
        self.db = MongoMessageDB()
        self.memory = ConversationMemory()
        self.attention = SemanticAttention()
        logger.info("[QA_SERVICE] Initialization complete")

    def _is_explicit_variable_assignment(self, question: str) -> bool:
        """Check if this is an EXPLICIT variable assignment like 'A = 55' or 'let x = 100'."""
        q = question.strip()
        return bool(re.match(r'^(?:let\s+)?([a-zA-Z_]\w*)\s*=\s*(.+)$', q, re.IGNORECASE))

    def _is_explicit_variable_recall(self, question: str) -> bool:
        """Check if this is EXPLICITLY asking about a defined variable."""
        q = question.lower().strip()
        
        match = re.search(r'(?:what(?:\'?s| is| was)|value\s+of|tell me)\s+([a-zA-Z_]\w*)', q)
        if match:
            var_name = match.group(1).lower()
            if len(var_name) == 1 or var_name.startswith('_'):
                logger.info(f"[VAR_CHECK] '{var_name}' detected as explicit variable")
                return True
            else:
                logger.info(f"[VAR_CHECK] '{var_name}' is a common word, not a variable")
                return False
        return False

    def _detect_user_statement(self, question: str, chat_id: str) -> Optional[Tuple[str, str]]:
        """Enhanced detection of user statements - ONLY name/age, NOT variable assignments."""
        q = question.strip()
        logger.info(f"[STATEMENT_DETECT] Checking: '{q}'")
        
        # Check for variable assignment first
        if self._is_explicit_variable_assignment(q):
            match = re.match(r'^(?:let\s+)?([a-zA-Z_]\w*)\s*=\s*(.+)$', q, re.IGNORECASE)
            if match:
                var_name = match.group(1).lower()
                value = match.group(2).strip()
                self.memory.set_fact(chat_id, f"var_{var_name}", value)
                response = f"Got it! I've noted that {var_name} = {value}."
                logger.info(f"[STATEMENT_DETECT] Variable assignment: {response}")
                return ("variable", response)
        
        # Pattern 1: "my name is X and age is Y"
        match = re.search(
            r'(?:my\s+)?name\s+is\s+([a-zA-Z]+)(?:\s+and\s+(?:my\s+)?age\s+is\s+(\d+))?',
            q,
            re.IGNORECASE
        )
        if match:
            name = match.group(1).strip()
            age = match.group(2).strip() if match.group(2) else None
            
            self.memory.set_fact(chat_id, "user_name", name)
            response_parts = [f"Nice to meet you, {name}!"]
            
            if age:
                self.memory.set_fact(chat_id, "user_age", age)
                response_parts.append(f"And I've noted that you're {age} years old.")
            
            response = " ".join(response_parts)
            logger.info(f"[STATEMENT_DETECT] Name statement detected: {response}")
            return ("name_age", response)
        
        # Pattern 2: "my age is X"
        match = re.search(
            r'(?:my\s+age\s+is|i\s+am|i\'m)\s+(\d+)(?:\s+years?\s+old)?',
            q,
            re.IGNORECASE
        )
        if match:
            age = match.group(1).strip()
            self.memory.set_fact(chat_id, "user_age", age)
            response = f"Got it! You're {age} years old."
            logger.info(f"[STATEMENT_DETECT] Age statement detected: {response}")
            return ("age", response)
        
        # Pattern 3: "my name is X"
        match = re.search(r'(?:my\s+)?name\s+is\s+([a-zA-Z]+)', q, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            self.memory.set_fact(chat_id, "user_name", name)
            response = f"Nice to meet you, {name}!"
            logger.info(f"[STATEMENT_DETECT] Name statement detected: {response}")
            return ("name", response)
        
        logger.info("[STATEMENT_DETECT] No statement pattern matched")
        return None

    def _detect_recall_question(self, question: str, chat_id: str) -> Optional[str]:
        """Enhanced recall detection - ONLY for explicit user facts."""
        q = question.lower().strip()
        logger.info(f"[RECALL_DETECT] Checking: '{q}'")
        
        # Pattern 1: "what is my name"
        if re.search(r"(?:what(?:'?s| is| was)|tell me|do you know)\s+my\s+name|who\s+am\s+i", q):
            name = self.memory.get_fact(chat_id, "user_name")
            if name:
                response = f"Your name is {name}."
                logger.info(f"[RECALL_DETECT] Name recall: {response}")
                return response
            else:
                response = "You haven't told me your name yet."
                logger.info(f"[RECALL_DETECT] Name not found: {response}")
                return response
        
        # Pattern 2: "what is my age"
        if re.search(r"(?:what(?:'?s| is| was)|tell me)\s+my\s+age|how\s+old\s+am\s+i", q):
            age = self.memory.get_fact(chat_id, "user_age")
            if age:
                response = f"You're {age} years old."
                logger.info(f"[RECALL_DETECT] Age recall: {response}")
                return response
            else:
                response = "You haven't told me your age yet."
                logger.info(f"[RECALL_DETECT] Age not found: {response}")
                return response
        
        # Pattern 3: EXPLICIT variable recall
        if self._is_explicit_variable_recall(q):
            match = re.search(r'(?:what(?:\'?s| is| was)|value\s+of|tell me)\s+([a-zA-Z_]\w*)', q)
            if match:
                var_name = match.group(1).lower()
                value = self.memory.get_fact(chat_id, f"var_{var_name}")
                if value:
                    response = f"The value of {var_name} is {value}."
                    logger.info(f"[RECALL_DETECT] Variable recall: {response}")
                    return response
                else:
                    response = f"I don't have a value stored for {var_name}."
                    logger.info(f"[RECALL_DETECT] Variable not found: {response}")
                    return response
        
        logger.info("[RECALL_DETECT] No recall pattern matched")
        return None

    def _detect_pronoun_reference(self, question: str, chat_id: str) -> Optional[str]:
        """Detect pronoun references to previous topics."""
        q = question.lower().strip()
        
        pronoun_patterns = [
            r'^(?:explain|what is|tell me about|describe)\s+(?:it|that|this)(?:\s|$|\?)',
            r'^(?:it|that|this)(?:\s|$|\?)',
            r'^(?:more about|details on)\s+(?:it|that|this)(?:\s|$|\?)'
        ]
        
        for pattern in pronoun_patterns:
            if re.search(pattern, q):
                last_topic = self.memory.get_last_topic(chat_id)
                if last_topic:
                    logger.info(f"[PRONOUN] Detected reference to: {last_topic}")
                    return last_topic
        
        return None

    def _extract_topic_from_question(self, question: str) -> Optional[str]:
        """Extract main topic from question."""
        q = question.lower()
        
        patterns = [
            r'(?:what is|what\'s|whats)\s+([a-zA-Z][a-zA-Z0-9\s]+?)(?:\?|$)',
            r'(?:explain|describe)\s+([a-zA-Z][a-zA-Z0-9\s]+?)(?:\?|$)',
            r'(?:tell me about|help me with|learn about)\s+([a-zA-Z][a-zA-Z0-9\s]+?)(?:\?|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, q)
            if match:
                topic = match.group(1).strip()
                logger.info(f"[TOPIC_EXTRACT] Extracted: '{topic}'")
                return topic
        
        return None

    def _is_off_topic(self, question: str) -> bool:
        """Detect off-topic questions."""
        q = question.lower()
        logger.info(f"[OFF_TOPIC] Checking: '{q}'")
        
        tech_keywords = [
            'program', 'code', 'algorithm', 'data', 'software', 'web', 'api', 'database',
            'python', 'java', 'javascript', 'react', 'node', 'machine learning', 'ai',
            'neural', 'model', 'system', 'design', 'function', 'class', 'variable',
            'framework', 'library', 'server', 'client', 'frontend', 'backend', 'stack',
            'computer', 'engineering', 'science', 'math', 'calculate', 'solve', 'explain',
            'what is', 'how to', 'why does', 'concept', 'theory', 'project', 'develop',
            'array', 'loop', 'string', 'object', 'recursion', 'sorting', 'search',
            'language', 'syntax', 'compiler', 'interpreter', 'debugging', 'testing'
        ]
        
        if any(kw in q for kw in tech_keywords):
            logger.info("[OFF_TOPIC] Technical keyword - ON TOPIC")
            return False
        
        offtopic_keywords = [
            'cafe racer', 'motorcycle', 'bike', 'motor', 'vehicle', 'car',
            'recipe', 'cook', 'food', 'restaurant', 'sports', 'football',
            'music', 'movie', 'fashion', 'travel', 'weather', 'politics'
        ]
        
        if any(kw in q for kw in offtopic_keywords):
            logger.info("[OFF_TOPIC] Off-topic keyword - OFF TOPIC")
            return True
        
        return False

    def _is_greeting(self, question: str) -> bool:
        """Detect greetings."""
        q = question.lower().strip()
        greetings = ['hi', 'hello', 'hey', 'yo', 'sup', "what's up", 'good morning']
        return q in greetings or (any(q.startswith(g) for g in greetings) and len(q.split()) <= 4)

    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context from chunks."""
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
        
        logger.info(f"[CONTEXT] Built with {len(context_parts)} chunks, {total_chars} chars")
        return "\n\n---\n\n".join(context_parts)

    def _has_material(self, user_id: str, material_id: Optional[str]) -> bool:
        """Check if user has material."""
        try:
            chunks = self.doc_processor.get_relevant_chunks(
                query="test", user_id=user_id, material_id=material_id,
                use_all_materials=(material_id is None), top_k=1
            )
            return bool(chunks)
        except:
            return False

    def get_recent_messages(self, chat_id: str, limit: int = MAX_HISTORY_MESSAGES) -> List:
        """Load conversation history."""
        try:
            msgs = self.db.get_last_messages(chat_id, limit)
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" 
                else AIMessage(content=m["content"])
                for m in msgs
            ]
            logger.info(f"[HISTORY] Loaded {len(history)} messages")
            return history
        except Exception as e:
            logger.exception(f"[HISTORY] Error: {e}")
            return []

    def answer_question(
        self,
        question: str,
        user_id: str,
        chat_id: str,
        material_id: Optional[str],
        general_mode: bool
    ) -> Dict:
        """Smart QA with semantic attention."""
        start_ts = time.time()
        logger.info(f"[QA] ========== NEW QUERY ==========")
        logger.info(f"[QA] Question: '{question}'")
        
        try:
            if not all([question, user_id, chat_id]):
                return {"answer": "Missing required fields", "sources": [], "mode": "error"}

            # Check for user statements
            statement_result = self._detect_user_statement(question, chat_id)
            if statement_result:
                _, answer = statement_result
                self.db.add_message(chat_id, user_id, "assistant", answer)
                return {"answer": answer, "sources": [], "mode": "statement"}

            # Check for recall questions
            recall_answer = self._detect_recall_question(question, chat_id)
            if recall_answer:
                self.db.add_message(chat_id, user_id, "assistant", recall_answer)
                return {"answer": recall_answer, "sources": [], "mode": "recall"}

            # Check for pronoun reference
            pronoun_topic = self._detect_pronoun_reference(question, chat_id)
            if pronoun_topic:
                question = f"{question} (referring to {pronoun_topic})"

            # Extract and store topic
            topic = self._extract_topic_from_question(question)
            if topic:
                self.memory.set_last_topic(chat_id, topic)

            # Load history
            full_history = self.get_recent_messages(chat_id, limit=MAX_HISTORY_MESSAGES)

            # GENERAL MODE
            if general_mode:
                logger.info("[QA] GENERAL mode")
                relevant_history = self.attention.select_relevant_context(question, full_history)
                
                facts = self.memory.get_all_facts(chat_id)
                facts_context = ""
                if facts:
                    facts_list = [f"{k.replace('var_', '').replace('user_', '')}: {v}" for k, v in facts.items()]
                    facts_context = f"\n\nUser context:\n" + "\n".join(facts_list)
                
                prompt = f"You are a helpful AI assistant. Answer naturally and concisely.{facts_context}"
                messages = [SystemMessage(content=prompt)] + relevant_history + [HumanMessage(content=question)]
                
                response = self.llm.invoke(messages)
                answer = response.content.strip()
                
                self.db.add_message(chat_id, user_id, "assistant", answer)
                return {"answer": answer, "sources": [], "mode": "general"}

            # Check for material
            has_material = self._has_material(user_id, material_id)
            
            if not has_material:
                logger.info("[QA] No material - expert knowledge")
                relevant_history = self.attention.select_relevant_context(question, full_history)
                
                facts = self.memory.get_all_facts(chat_id)
                facts_context = ""
                if facts:
                    facts_list = [f"{k.replace('var_', '').replace('user_', '')}: {v}" for k, v in facts.items()]
                    facts_context = f"\n\nUser context:\n" + "\n".join(facts_list)
                
                prompt = f"You are a helpful AI tutor specializing in CS and programming.{facts_context}"
                messages = [SystemMessage(content=prompt)] + relevant_history + [HumanMessage(content=question)]
                
                response = self.llm.invoke(messages)
                answer = response.content.strip()
                
                self.db.add_message(chat_id, user_id, "assistant", answer)
                return {"answer": answer, "sources": [], "mode": "no_material"}

            # MATERIAL MODE
            logger.info("[QA] MATERIAL mode")
            
            if self._is_greeting(question):
                answer = "Hello! I'm here to help you with your studies."
                self.db.add_message(chat_id, user_id, "assistant", answer)
                return {"answer": answer, "sources": [], "mode": "greeting"}

            if self._is_off_topic(question):
                answer = "That's outside my scope. I'm here for CS/programming topics."
                self.db.add_message(chat_id, user_id, "assistant", answer)
                return {"answer": answer, "sources": [], "mode": "off_topic"}

            # Retrieve chunks
            chunks = self.doc_processor.get_relevant_chunks(
                query=question,
                user_id=user_id,
                material_id=material_id,
                use_all_materials=(material_id is None),
                top_k=RETRIEVE_TOP_K
            ) or []
            
            context = self._build_context(chunks)
            max_similarity = max([c.get("similarity", 0) for c in chunks], default=0)
            has_relevant_material = bool(context and max_similarity > 0.2)
            
            # Select relevant history
            relevant_history = self.attention.select_relevant_context(question, full_history)
            
            # Add facts
            facts = self.memory.get_all_facts(chat_id)
            facts_context = ""
            if facts:
                facts_list = [f"{k.replace('var_', '').replace('user_', '')}: {v}" for k, v in facts.items()]
                facts_context = f"\n\nUser's stated facts:\n" + "\n".join(facts_list)
            
            # Build prompt
            if has_relevant_material:
                system_prompt = (
                    "You are a helpful AI tutor. Answer based on material and conversation history.\n\n"
                    "RULES:\n"
                    "1. Use history for context/pronouns\n"
                    "2. Be concise\n"
                    "3. Direct answers\n"
                    "4. Natural tone\n"
                    f"{facts_context}"
                )
                user_msg = f"Material:\n{context}\n\nQuestion: {question}"
            else:
                system_prompt = (
                    "You are a knowledgeable AI tutor in CS/programming.\n\n"
                    "RULES:\n"
                    "1. Use history for context\n"
                    "2. Be concise\n"
                    "3. Natural tone\n"
                    f"{facts_context}"
                )
                user_msg = f"Question: {question}"
            
            messages = [SystemMessage(content=system_prompt)] + relevant_history + [HumanMessage(content=user_msg)]
            
            response = self.llm.invoke(messages)
            answer = response.content.strip()
            
            self.db.add_message(chat_id, user_id, "assistant", answer)
            
            sources = []
            if has_relevant_material:
                for c in chunks[:5]:
                    sources.append({
                        "content": c.get("content", "")[:300],
                        "similarity": round(c.get("similarity", 0), 3),
                        "metadata": c.get("metadata", {})
                    })
            
            mode = "material" if has_relevant_material else "expert_knowledge"
            elapsed = time.time() - start_ts
            logger.info(f"[QA] Mode: {mode} | Time: {elapsed:.2f}s")
            return {"answer": answer, "sources": sources, "mode": mode}

        except Exception as e:
            logger.exception(f"[QA] ERROR: {e}")
            error_msg = "I encountered an error. Please try again."
            self.db.add_message(chat_id, user_id, "assistant", error_msg)
            return {"answer": error_msg, "sources": [], "mode": "error", "error": str(e)}