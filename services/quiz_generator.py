from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from services.document_processor import DocumentProcessor
import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CanonicalTopicMapper:
    """Maintains consistent topic naming across the application"""
    
    def __init__(self, llm):
        self.llm = llm
        # Cache to store user_id -> {raw_topic: canonical_topic}
        self.topic_cache = {}
    
    def get_canonical_topic(self, raw_topic: str, user_id: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Get or create a canonical topic name using LLM with strict consistency rules
        
        Returns:
            {
                'canonical_topic': str,
                'confidence': str,
                'is_new': bool,
                'matched_existing': Optional[str]
            }
        """
        # Initialize user cache if needed
        if user_id not in self.topic_cache:
            self.topic_cache[user_id] = {}
        
        user_topics = self.topic_cache[user_id]
        
        # Check exact match first (case-insensitive)
        raw_lower = raw_topic.lower().strip()
        for cached_raw, cached_canonical in user_topics.items():
            if cached_raw.lower().strip() == raw_lower:
                logger.info(f"Exact cache hit: '{raw_topic}' -> '{cached_canonical}'")
                return {
                    'canonical_topic': cached_canonical,
                    'confidence': 'high',
                    'is_new': False,
                    'matched_existing': cached_raw
                }
        
        # If we have existing topics, ask LLM to match or create new
        if user_topics:
            existing_topics = list(set(user_topics.values()))
            canonical = self._match_or_create_topic(raw_topic, existing_topics, context)
        else:
            # First topic for this user - create canonical version
            canonical = self._create_canonical_topic(raw_topic, context)
        
        # Cache the mapping
        user_topics[raw_topic] = canonical['canonical_topic']
        
        logger.info(f"Topic mapping: '{raw_topic}' -> '{canonical['canonical_topic']}' "
                   f"(new={canonical.get('is_new', True)})")
        
        return canonical
    
    def _match_or_create_topic(self, raw_topic: str, existing_topics: List[str], context: Optional[str]) -> Dict[str, Any]:
        """Use LLM to match against existing topics or create new canonical name"""
        try:
            context_info = ""
            if context:
                context_info = f"\n\n**Content Context** (first 800 chars):\n{context[:800]}"
            
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=f"""You are a topic consistency expert for an educational system.

**Your Critical Task**: Determine if the user's input refers to the SAME topic as any existing topic, or if it's a NEW topic.

**Existing Canonical Topics in System**:
{json.dumps(existing_topics, indent=2)}

**Matching Rules** (VERY IMPORTANT):
1. **Same Core Concept = Same Topic**: 
   - "Recursion", "Recursion in Computer Science", "Recursion Algorithm" → ALL map to "Recursion"
   - "Insertion Sort", "Insertion Sort Algorithm" → ALL map to "Insertion Sort"
   - "Bubble Sort", "Bubble Sort Algorithm" → ALL map to "Bubble Sort"

2. **Only Create New Topic If**:
   - The concept is fundamentally different (e.g., "Recursion" vs "Dynamic Programming")
   - It's a specific subtopic that warrants separate tracking (e.g., "Recursion" vs "Tail Recursion Optimization")

3. **Canonical Format**:
   - Use the SHORTEST, most common form
   - Title Case (e.g., "Recursion", "Insertion Sort", "Binary Search Trees")
   - NO unnecessary descriptors (e.g., "Algorithm", "in Computer Science", "Concept")
   - Be consistent: if existing topic is "Insertion Sort", use that exactly

4. **Spelling Corrections**:
   - Fix obvious typos but preserve the core concept
   - "Recurson" → matches "Recursion"
   - "Insrtion Sort" → matches "Insertion Sort"

**Decision Priority**:
1. Match existing topic if >80% conceptual overlap
2. Only create new if truly different concept
3. When in doubt, match existing (prefer consistency)

Return ONLY valid JSON:
{{
  "canonical_topic": "Exact Match from Existing OR New Canonical Name",
  "confidence": "high/medium/low",
  "is_new": true/false,
  "matched_existing": "existing topic name if matched, null if new",
  "reasoning": "Brief explanation of decision"
}}"""),
                HumanMessage(content=f"""**User Input**: {raw_topic}{context_info}

Determine if this matches an existing topic or needs a new canonical name.""")
            ])
            
            messages = prompt.format_messages()
            response = self.llm.invoke(messages)
            content = self._clean_json_response(response.content)
            result = json.loads(content)
            
            # Validate that matched_existing actually exists in our list
            if not result.get('is_new') and result.get('matched_existing'):
                if result['matched_existing'] not in existing_topics:
                    logger.warning(f"LLM returned non-existent match: {result['matched_existing']}")
                    # Try to find closest match
                    for topic in existing_topics:
                        if topic.lower() == result['matched_existing'].lower():
                            result['matched_existing'] = topic
                            result['canonical_topic'] = topic
                            break
                    else:
                        # No match found, treat as new
                        result['is_new'] = True
                        result['matched_existing'] = None
            
            return result
            
        except Exception as e:
            logger.error(f"Topic matching failed: {e}. Creating new canonical topic.")
            return self._create_canonical_topic(raw_topic, context)
    
    def _create_canonical_topic(self, raw_topic: str, context: Optional[str]) -> Dict[str, Any]:
        """Create a new canonical topic name"""
        try:
            context_info = ""
            if context:
                context_info = f"\n\n**Content Context** (first 800 chars):\n{context[:800]}"
            
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a topic naming expert for an educational system.

**Task**: Create a canonical topic name that will be used consistently.

**Canonical Format Rules**:
1. **Short and Simple**: Use the most common, concise form
2. **Title Case**: Proper capitalization (e.g., "Binary Search Trees")
3. **No Unnecessary Words**: 
   - Remove: "Algorithm", "Concept", "in Computer Science", "Topic"
   - Keep: Core concept only
4. **Examples**:
   - ❌ "Recursion Algorithm in Computer Science"
   - ✅ "Recursion"
   - ❌ "Insertion Sort Algorithm"
   - ✅ "Insertion Sort"
   - ❌ "Binary Search Tree Data Structure"
   - ✅ "Binary Search Trees"

5. **Fix Spelling**: Correct any typos
6. **Standard Terminology**: Use widely accepted terms

Return ONLY valid JSON:
{{
  "canonical_topic": "Canonical Topic Name",
  "confidence": "high/medium/low",
  "is_new": true,
  "matched_existing": null,
  "reasoning": "Brief explanation"
}}"""),
                HumanMessage(content=f"""**User Input**: {raw_topic}{context_info}

Create a canonical topic name.""")
            ])
            
            messages = prompt.format_messages()
            response = self.llm.invoke(messages)
            content = self._clean_json_response(response.content)
            result = json.loads(content)
            result['is_new'] = True
            result['matched_existing'] = None
            
            return result
            
        except Exception as e:
            logger.error(f"Canonical topic creation failed: {e}. Using sanitized input.")
            # Fallback: basic sanitization
            sanitized = raw_topic.strip().title()
            # Remove common suffixes
            for suffix in [' Algorithm', ' Concept', ' Topic', ' In Computer Science']:
                if sanitized.endswith(suffix):
                    sanitized = sanitized[:-len(suffix)].strip()
            
            return {
                'canonical_topic': sanitized,
                'confidence': 'low',
                'is_new': True,
                'matched_existing': None,
                'reasoning': 'Fallback sanitization due to error'
            }
    
    def _clean_json_response(self, content: str) -> str:
        """Clean LLM response to extract valid JSON"""
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        return content.strip()


class QuizGenerator:
    """Enterprise-grade quiz generation system with canonical topic mapping"""
    
    def __init__(self):
        # Initialize Azure OpenAI LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_API_NAME"),
            model_name="gpt-4o",
            temperature=0.7
        )
        
        self.doc_processor = DocumentProcessor()
        self.topic_mapper = CanonicalTopicMapper(self.llm)
        
        # Enhanced difficulty configurations
        self.difficulty_config = {
            'easy': {
                'description': 'Focus on basic concepts, definitions, and recall.',
                'cognitive_levels': ['Remember', 'Understand'],
                'complexity': 'Simple scenarios with direct application',
                'temperature': 0.5
            },
            'medium': {
                'description': 'Include application and understanding questions.',
                'cognitive_levels': ['Understand', 'Apply', 'Analyze'],
                'complexity': 'Moderate scenarios requiring reasoning',
                'temperature': 0.7
            },
            'hard': {
                'description': 'Focus on analysis, synthesis, evaluation.',
                'cognitive_levels': ['Analyze', 'Evaluate', 'Create'],
                'complexity': 'Complex scenarios with edge cases',
                'temperature': 0.8
            }
        }
    
    def _assess_content_relevance(self, topic: str, content: str) -> Dict[str, Any]:
        """Use LLM to assess if the retrieved content is relevant to the topic"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a content relevance analyzer.

Determine if the content contains sufficient information about the topic.

Return ONLY valid JSON:
{
  "relevance": "high/medium/low/none",
  "confidence": "high/medium/low",
  "reasoning": "Brief explanation",
  "use_material": true/false,
  "coverage_percentage": 0-100
}

Set use_material=true only if relevance is high/medium with 40%+ coverage."""),
                HumanMessage(content=f"""**Topic**: {topic}

**Content** (first 2000 chars):
{content[:2000]}

Assess relevance.""")
            ])
            
            messages = prompt.format_messages()
            response = self.llm.invoke(messages)
            content_str = self._clean_json_response(response.content)
            assessment = json.loads(content_str)
            
            logger.info(f"Content relevance for '{topic}': {assessment['relevance']} "
                       f"(coverage: {assessment.get('coverage_percentage', 0)}%)")
            
            return assessment
            
        except Exception as e:
            logger.warning(f"Relevance assessment failed: {e}")
            return {
                "relevance": "low",
                "confidence": "low",
                "reasoning": "Assessment failed",
                "use_material": False,
                "coverage_percentage": 0
            }
    
    def _get_material_context_with_relevance_check(
        self,
        topic: str,
        user_id: str,
        material_id: Optional[str] = None,
        use_all_materials: bool = False,
        top_k: int = 10
    ) -> Tuple[Optional[List[Dict]], Optional[str], str, Dict[str, Any]]:
        """Retrieve material chunks and assess relevance"""
        
        if use_all_materials:
            logger.info(f"Using global AI knowledge for topic: {topic}")
            return None, None, 'global_knowledge', {
                'decision': 'user_requested_global',
                'reasoning': 'User explicitly requested global AI knowledge'
            }
        
        try:
            if material_id:
                logger.info(f"Retrieving from material: {material_id}")
                chunks = self.doc_processor.get_relevant_chunks(
                    query=topic,
                    user_id=user_id,
                    material_id=material_id,
                    use_all_materials=False,
                    top_k=top_k
                )
            else:
                raise ValueError("Either material_id or use_all_materials must be specified")
            
            if not chunks or len(chunks) == 0:
                logger.warning(f"No chunks found for topic: {topic}")
                return None, None, 'global_knowledge', {
                    'decision': 'no_content_found',
                    'reasoning': 'No relevant content found'
                }
            
            # Build context
            context_parts = []
            for chunk in chunks:
                source = chunk.get('material_name', 'Unknown')
                content = chunk['content']
                context_parts.append(f"[Source: {source}]\n{content}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Assess relevance
            relevance_assessment = self._assess_content_relevance(topic, context)
            
            if relevance_assessment.get('use_material', False):
                logger.info(f"Using {len(chunks)} chunks from materials")
                return chunks, context, 'specific_material', {
                    'decision': 'using_material',
                    'relevance': relevance_assessment['relevance'],
                    'coverage': relevance_assessment.get('coverage_percentage', 0),
                    'reasoning': relevance_assessment['reasoning']
                }
            else:
                logger.warning(f"Material relevance too low, using global knowledge")
                return None, None, 'global_knowledge', {
                    'decision': 'low_relevance_fallback',
                    'relevance': relevance_assessment['relevance'],
                    'coverage': relevance_assessment.get('coverage_percentage', 0),
                    'reasoning': relevance_assessment['reasoning']
                }
                
        except Exception as e:
            logger.error(f"Error retrieving material: {e}")
            return None, None, 'global_knowledge', {
                'decision': 'error_fallback',
                'reasoning': f'Error: {str(e)}'
            }
    
    def _clean_json_response(self, content: str) -> str:
        """Clean LLM response to extract valid JSON"""
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        return content.strip()
    
    def _validate_quiz_questions(self, questions: List[Dict]) -> List[Dict]:
        """Validate quiz questions"""
        validated = []
        
        for i, q in enumerate(questions):
            try:
                if not all(key in q for key in ['question', 'options', 'correct_answer', 'explanation']):
                    logger.warning(f"Question {i+1} missing required fields")
                    continue
                
                if not isinstance(q['options'], dict) or len(q['options']) != 4:
                    logger.warning(f"Question {i+1} invalid options")
                    continue
                
                if q['correct_answer'] not in q['options']:
                    logger.warning(f"Question {i+1} invalid correct_answer")
                    continue
                
                q['question'] = q['question'].strip()
                if len(q['question']) < 10:
                    logger.warning(f"Question {i+1} too short")
                    continue
                
                validated.append(q)
                
            except Exception as e:
                logger.error(f"Error validating question {i+1}: {e}")
                continue
        
        return validated
    
    def generate_quiz(
        self,
        topic: str,
        user_id: str,
        material_id: Optional[str] = None,
        num_questions: int = 5,
        difficulty: str = 'medium',
        use_all_materials: bool = False
    ) -> Dict[str, Any]:
        """Generate quiz with canonical topic naming"""
        try:
            # Validate inputs
            if not topic or not user_id:
                raise ValueError("Topic and user_id required")
            
            if material_id and use_all_materials:
                raise ValueError("Cannot specify both material_id and use_all_materials")
            
            if not (material_id or use_all_materials):
                raise ValueError("Must specify either material_id or use_all_materials")
            
            difficulty = difficulty.lower()
            if difficulty not in self.difficulty_config:
                logger.warning(f"Invalid difficulty, defaulting to medium")
                difficulty = 'medium'
            
            num_questions = max(3, min(num_questions, 20))
            
            # Retrieve material context
            chunks, context, source_type, relevance_info = self._get_material_context_with_relevance_check(
                topic, user_id, material_id, use_all_materials, top_k=15
            )
            
            # ✨ GET CANONICAL TOPIC NAME
            topic_result = self.topic_mapper.get_canonical_topic(topic, user_id, context)
            canonical_topic = topic_result['canonical_topic']
            
            # Get difficulty config
            diff_config = self.difficulty_config[difficulty]
            
            # Build prompt based on actual source being used
            if source_type == 'global_knowledge':
                # Global AI Knowledge prompt
                system_content = f"""You are an expert educational assessment designer with deep pedagogical knowledge.

Your task: Generate {num_questions} high-quality multiple-choice questions based on your comprehensive knowledge of the topic.

**Topic**: {canonical_topic}
**Difficulty Level**: {difficulty.upper()}
**Cognitive Focus**: {', '.join(diff_config['cognitive_levels'])}
**Source**: Global AI Knowledge

**Difficulty Guidelines**:
{diff_config['description']}

**Complexity**: {diff_config['complexity']}

**Quality Standards**:
1. Each question must have EXACTLY 4 options (A, B, C, D)
2. Only ONE correct answer per question
3. Distractors (wrong answers) should be plausible but clearly incorrect to knowledgeable students
4. Questions should test deep understanding, not just memorization
5. Avoid ambiguous phrasing, trick questions, or "all of the above" options
6. Include brief, educational explanations for correct answers
7. Use clear, professional language appropriate for the subject matter
8. Draw from authoritative, well-established knowledge in the field

**Question Distribution** (aim for variety):
- Conceptual understanding: 40%
- Application/problem-solving: 40%
- Analysis/evaluation: 20%

Return ONLY a valid JSON array with this EXACT structure:
[
  {{
    "question": "Clear, specific question text ending with a question mark?",
    "options": {{
      "A": "First option text",
      "B": "Second option text",
      "C": "Third option text",
      "D": "Fourth option text"
    }},
    "correct_answer": "A",
    "explanation": "Concise explanation (2-3 sentences) of why this answer is correct and why it matters."
  }}
]

CRITICAL: Return ONLY the JSON array, no additional text, no preamble, no markdown formatting."""
                
                human_content = f"""**Topic for Assessment**: {canonical_topic}

**Note**: {relevance_info.get('reasoning', 'Using global AI knowledge')}

**Task**: Generate {num_questions} {difficulty}-level multiple-choice questions using your comprehensive knowledge of this topic. Focus on creating questions that assess true understanding and practical application of core concepts."""
            
            else:
                # Material-based prompt
                system_content = f"""You are an expert educational assessment designer with deep pedagogical knowledge.

Your task: Generate {num_questions} high-quality multiple-choice questions based STRICTLY on the provided content.

**Topic**: {canonical_topic}
**Difficulty Level**: {difficulty.upper()}
**Cognitive Focus**: {', '.join(diff_config['cognitive_levels'])}
**Source**: User-provided study materials

**Difficulty Guidelines**:
{diff_config['description']}

**Complexity**: {diff_config['complexity']}

**Quality Standards**:
1. Each question must have EXACTLY 4 options (A, B, C, D)
2. Only ONE correct answer per question
3. ALL questions MUST be directly answerable from the provided content
4. Do NOT use external knowledge - stick to the material provided
5. Distractors should be plausible but clearly incorrect based on the content
6. Questions should test deep understanding, not just memorization
7. Avoid ambiguous phrasing, trick questions, or "all of the above" options
8. Include brief explanations referencing the provided content

**Question Distribution** (aim for variety):
- Conceptual understanding: 40%
- Application/problem-solving: 40%
- Analysis/evaluation: 20%

Return ONLY a valid JSON array with this EXACT structure:
[
  {{
    "question": "Clear, specific question text ending with a question mark?",
    "options": {{
      "A": "First option text",
      "B": "Second option text",
      "C": "Third option text",
      "D": "Fourth option text"
    }},
    "correct_answer": "A",
    "explanation": "Concise explanation (2-3 sentences) of why this answer is correct based on the material."
  }}
]

CRITICAL: Return ONLY the JSON array, no additional text, no preamble, no markdown formatting."""
                
                human_content = f"""**Content Source**: {chunks[0].get('material_name', 'Study Materials')}

**Topic for Assessment**: {canonical_topic}

**Reference Material**:
{context}

**Task**: Generate {num_questions} {difficulty}-level multiple-choice questions that assess understanding of THIS SPECIFIC MATERIAL. Do not use external knowledge."""
            
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_content),
                HumanMessage(content=human_content)
            ])
            
            # Adjust temperature
            temp_llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_deployment=os.getenv("AZURE_OPENAI_API_NAME"),
                model_name="gpt-4o",
                temperature=diff_config['temperature']
            )
            
            messages = prompt.format_messages()
            response = temp_llm.invoke(messages)
            
            # Log raw response for debugging
            logger.debug(f"Raw LLM response (first 500 chars): {response.content[:500]}")
            
            content_str = self._clean_json_response(response.content)
            logger.debug(f"Cleaned JSON (first 500 chars): {content_str[:500]}")
            
            quiz_data = json.loads(content_str)
            logger.debug(f"Parsed quiz_data type: {type(quiz_data)}, length: {len(quiz_data) if isinstance(quiz_data, list) else 'N/A'}")
            
            validated_questions = self._validate_quiz_questions(quiz_data)
            
            if len(validated_questions) < num_questions * 0.7:
                raise ValueError(f"Only {len(validated_questions)} valid questions generated")
            
            logger.info(f"Generated {len(validated_questions)} questions for '{canonical_topic}'")
            
            metadata = {
                'generated_at': str(os.getenv('TIMESTAMP', '')),
                'source_type': source_type,
                'topic_was_new': topic_result.get('is_new', False),
                'topic_matched': topic_result.get('matched_existing'),
                'topic_confidence': topic_result.get('confidence'),
                'content_decision': relevance_info.get('decision'),
                'content_relevance': relevance_info.get('relevance')
            }
            
            if chunks:
                metadata['num_sources'] = len(set(c.get('material_name') for c in chunks))
            
            return {
                'topic': canonical_topic,  # ✨ Always use canonical
                'original_topic': topic,
                'difficulty': difficulty,
                'material_scope': source_type,
                'material_id': material_id,
                'questions': validated_questions[:num_questions],
                'metadata': metadata
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return {'error': 'Failed to parse quiz format', 'details': str(e)}
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return {'error': str(e)}
        except Exception as e:
            logger.error(f"Quiz generation failed: {e}", exc_info=True)
            return {'error': 'Unexpected error', 'details': str(e)}
    
    def generate_flashcards(
        self,
        topic: str,
        user_id: str,
        material_id: Optional[str] = None,
        num_cards: int = 10,
        use_all_materials: bool = False
    ) -> Dict[str, Any]:
        """Generate flashcards with canonical topic naming"""
        try:
            chunks, context, source_type, relevance_info = self._get_material_context_with_relevance_check(
                topic, user_id, material_id, use_all_materials, top_k=15
            )
            
            # ✨ GET CANONICAL TOPIC NAME
            topic_result = self.topic_mapper.get_canonical_topic(topic, user_id, context)
            canonical_topic = topic_result['canonical_topic']
            
            # Build prompts (use your existing logic)
            if source_type == 'global_knowledge':
                system_content = f"""Generate {num_cards} flashcards on {canonical_topic}.
Return JSON array with front/back."""
                human_content = f"Topic: {canonical_topic}\nGenerate flashcards."
            else:
                system_content = f"""Generate {num_cards} flashcards from provided content only.
Return JSON array."""
                human_content = f"Topic: {canonical_topic}\n\nContent:\n{context}"
            
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_content),
                HumanMessage(content=human_content)
            ])
            
            messages = prompt.format_messages()
            response = self.llm.invoke(messages)
            content_str = self._clean_json_response(response.content)
            flashcards = json.loads(content_str)
            
            validated = [
                card for card in flashcards
                if 'front' in card and 'back' in card
                and len(card['front'].strip()) > 5
                and len(card['back'].strip()) > 10
            ]
            
            logger.info(f"Generated {len(validated)} flashcards for '{canonical_topic}'")
            
            return {
                'topic': canonical_topic,  # ✨ Always use canonical
                'original_topic': topic,
                'flashcards': validated[:num_cards],
                'metadata': {
                    'topic_was_new': topic_result.get('is_new'),
                    'topic_matched': topic_result.get('matched_existing'),
                    'topic_confidence': topic_result.get('confidence'),
                    'source_type': source_type,
                    'content_decision': relevance_info.get('decision')
                }
            }
            
        except Exception as e:
            logger.error(f"Flashcard generation failed: {e}")
            return {'error': str(e)}