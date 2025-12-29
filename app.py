from flask import Flask, request, jsonify
from flask_cors import CORS
from services.document_processor import DocumentProcessor
from services.qa_service import QAService
from services.quiz_generator import QuizGenerator
from services.socratic_tutor import SocraticTutor
import os
import tempfile
import base64
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

#Initialize
doc_processor = DocumentProcessor()
qa_service = QAService()
socratic_tutor = SocraticTutor()
quiz_generator = QuizGenerator()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/process-document', methods=['POST'])
def process_document():
    try:
        data = request.get_json(force=True)
        file_buffer = data.get('file_buffer')
        file_name = data.get('fiale_name')
        user_id = data.get('user_id')
        material_id = data.get('material_id')

        if not file_buffer:
            return jsonify({"success": False, "error": "Missing file_buffer"}), 400

        # Decode file
        file_bytes = base64.b64decode(file_buffer)

        # Create temp file
        suffix = os.path.splitext(file_name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        try:
            result = doc_processor.process_document(
                temp_path, user_id, material_id
            )
            return jsonify(result), 200
        finally:
            # DELETE FILE AFTER PROCESSING
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        print("Error in /process-document:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/ask-question', methods=['POST'])
def ask_question():
    
    try:
        data = request.json
        question = data.get('question')
        user_id = data.get('userId')
        chat_id = data.get('chatId')
        material_id = data.get('materialId')
        use_all_materials = data.get('useAllMaterials', False)

        if not question or not user_id or not chat_id:
            return jsonify({"error": "Missing required fields"}), 400

        result = qa_service.answer_question(
            question,
            user_id,
            chat_id,
            material_id,
            use_all_materials
        )

        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"--error": str(e)}), 500

@app.route('/socratic-question', methods=['POST'])
def socratic_question():
    
    """Socratic questioning mode"""
    try:
        data = request.json
        question = data.get('question')
        user_id = data.get('userId')
        material_id = data.get('materialId')
        chat_id = data.get('chatId')
        use_all_materials = data.get('useAllMaterials', False)
        
        response = socratic_tutor.generate_questions(
            question,
            user_id,
            chat_id,
            material_id,
            use_all_materials
        )
        return jsonify({"questions": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    
    """
    Generate quiz from materials with enterprise-grade validation
    
    Request body:
    {
        "topic": str,                    # User's topic (may have typos)
        "user_id": str,                  # User identifier
        "material_id": str | null,       # Specific material (mutually exclusive with use_all_materials)
        "num_questions": int,            # Number of questions (3-20)
        "difficulty": str,               # 'easy', 'medium', or 'hard'
        "use_all_materials": bool        # Use all materials instead of specific one
    }
    
    Returns:
    {
        "quiz": {
            "topic": str,                # Normalized topic
            "original_topic": str,       # User's original input
            "difficulty": str,
            "questions": [...],
            "metadata": {...}
        }
    }
    """
    try:
        data = request.json
        
        # Extract parameters
        topic = data.get('topic')
        user_id = data.get('user_id')
        material_id = data.get('material_id')
        num_questions = data.get('num_questions', 5)
        difficulty = data.get('difficulty', 'medium')
        use_all_materials = data.get('use_all_materials', False)
        
        # Validate required fields
        if not topic:
            return jsonify({
                "error": "Missing required field: 'topic'"
            }), 400
        
        if not user_id:
            return jsonify({
                "error": "Missing required field: 'user_id'"
            }), 400
        
        # Validate material scope
        if material_id and use_all_materials:
            return jsonify({
                "error": "Cannot specify both 'material_id' and 'use_all_materials'. Choose one."
            }), 400
        
        if not material_id and not use_all_materials:
            return jsonify({
                "error": "Must specify either 'material_id' or 'use_all_materials'=true"
            }), 400
        
        logger.info(
            f"Quiz generation request: user={user_id}, topic='{topic}', "
            f"material={'specific' if material_id else 'all'}, "
            f"questions={num_questions}, difficulty={difficulty}"
        )
        
        # Generate quiz
        quiz = quiz_generator.generate_quiz(
            topic=topic,
            user_id=user_id,
            material_id=material_id,
            num_questions=num_questions,
            difficulty=difficulty,
            use_all_materials=use_all_materials
        )
        
        # Check for errors
        if 'error' in quiz:
            logger.warning(f"Quiz generation failed: {quiz['error']}")
            return jsonify(quiz), 400
        
        logger.info(
            f"Quiz generated successfully: {len(quiz['questions'])} questions, "
            f"topic: '{quiz['topic']}' (normalized from '{quiz['original_topic']}')"
        )
        
        return jsonify({"quiz": quiz}), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in generate_quiz: {e}", exc_info=True)
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e)
        }), 500

@app.route('/generate-flashcards', methods=['POST'])
def generate_flashcards():
    
    try:
        data = request.json
        topic = data.get('topic')
        user_id = data.get('user_id')
        material_id = data.get('material_id')
        num_cards = data.get('num_cards', 10)
        use_all_materials = data.get('use_all_materials', False)
        
        flashcards = quiz_generator.generate_flashcards(
            topic,
            user_id,
            material_id,
            num_cards,
            use_all_materials
        )
        return jsonify({"flashcards": flashcards}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)