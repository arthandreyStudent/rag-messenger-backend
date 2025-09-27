from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from typing import Dict, List, Any
import logging
from datetime import datetime

# Import your multilingual RAG chatbot and conversation memory
from rag_app import MultilingualRAGChatbot
from conversation_memory import ConversationMemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global instances (singleton pattern)
chatbot_instance = None
memory_manager = None


def get_chatbot():
    """Get or create the multilingual RAG chatbot instance"""
    global chatbot_instance
    if chatbot_instance is None:
        logger.info("Initializing Multilingual RAG Chatbot...")
        chatbot_instance = MultilingualRAGChatbot()
        if not chatbot_instance.is_initialized:
            logger.error("Failed to initialize Multilingual RAG Chatbot")
            raise RuntimeError("Multilingual RAG Chatbot initialization failed")
        logger.info(f"Multilingual RAG Chatbot initialized with {len(chatbot_instance.vector_db.chunks)} chunks")
        logger.info(f"Supported languages: {list(chatbot_instance.multilingual_handler.supported_languages.keys())}")
    return chatbot_instance


def get_memory_manager():
    """Get or create the conversation memory manager"""
    global memory_manager
    if memory_manager is None:
        logger.info("Initializing Conversation Memory Manager...")
        memory_manager = ConversationMemoryManager()
        logger.info("Conversation Memory Manager initialized")
    return memory_manager


@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint with memory status"""
    try:
        chatbot = get_chatbot()
        memory = get_memory_manager()
        memory_status = memory.health_check()

        return jsonify({
            "status": "healthy",
            "message": "Multilingual RAG Chatbot API is running",
            "chunks_count": len(chatbot.vector_db.chunks),
            "initialized": chatbot.is_initialized,
            "supported_languages": list(chatbot.multilingual_handler.supported_languages.keys()),
            "memory_status": memory_status,
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with persistent conversation memory"""
    try:
        # Get request data
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        message = data.get('message', '').strip()
        user_id = data.get('user_id', 'default')

        # For Facebook Messenger integration
        facebook_sender_id = data.get('sender', {}).get('id')
        if facebook_sender_id:
            user_id = f"fb_{facebook_sender_id}"

        if not message:
            return jsonify({"error": "No message provided"}), 400

        logger.info(f"Received message from user {user_id}: {message}")

        # Get instances
        chatbot = get_chatbot()
        memory = get_memory_manager()

        # Get conversation context for continuity
        conversation_context = memory.get_conversation_context(user_id)
        conversation_summary = memory.get_conversation_summary(user_id)

        # Generate response with context if available
        if hasattr(chatbot, 'generate_response_with_context') and conversation_context:
            result = chatbot.generate_response_with_context(
                message, conversation_context, conversation_summary
            )
        else:
            result = chatbot.generate_response(message)

        # Store user message in memory
        memory.add_message(
            user_id=user_id,
            role="user",
            content=message,
            language=result["language"],
            metadata={"source": "api", "endpoint": "chat"}
        )

        # Store assistant response in memory
        memory.add_message(
            user_id=user_id,
            role="assistant",
            content=result["response"],
            language=result["language"],
            metadata={
                "confidence": result["confidence"],
                "sources_count": len(result["sources"])
            }
        )

        # Get updated history for response
        updated_history = memory.get_conversation_history(user_id)

        logger.info(f"Generated contextual {result['language']} response for user {user_id}")

        return jsonify({
            "response": result["response"],
            "language": result["language"],
            "language_name": chatbot.multilingual_handler.supported_languages[result["language"]],
            "history": updated_history,
            "confidence": result["confidence"],
            "sources_count": len(result["sources"]),
            "has_context": bool(conversation_context),
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/chat/simple', methods=['POST'])
def chat_simple():
    """Simplified chat endpoint with basic multilingual info"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        message = data.get('message', '').strip()

        if not message:
            return jsonify({"error": "No message provided"}), 400

        logger.info(f"Simple chat request: {message}")

        # Get chatbot and generate response
        chatbot = get_chatbot()
        result = chatbot.generate_response(message)

        # Return response with language info
        return jsonify({
            "response": result["response"],
            "language": result["language"],  # ADDED: Language detection
            "confidence": result["confidence"]  # ADDED: Confidence score
        }), 200

    except Exception as e:
        logger.error(f"Error in simple chat endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/chat/detailed', methods=['POST'])
def chat_detailed():
    """Detailed chat endpoint with full multilingual RAG information"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        message = data.get('message', '').strip()
        user_id = data.get('user_id', 'default')

        if not message:
            return jsonify({"error": "No message provided"}), 400

        logger.info(f"Detailed chat request from user {user_id}: {message}")

        # Get chatbot and generate response
        chatbot = get_chatbot()
        result = chatbot.generate_response(message)

        # Return detailed multilingual information
        return jsonify({
            "response": result["response"],
            "language": result["language"],
            "language_name": chatbot.multilingual_handler.supported_languages[result["language"]],
            "confidence": result["confidence"],
            "sources": result["sources"],
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "llm_model": chatbot.llm_model,
                "embedding_model": chatbot.embedding_model,
                "chunks_total": len(chatbot.vector_db.chunks),
                "supported_languages": list(chatbot.multilingual_handler.supported_languages.keys())
            }
        }), 200

    except Exception as e:
        logger.error(f"Error in detailed chat endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages"""
    try:
        chatbot = get_chatbot()
        return jsonify({
            "supported_languages": chatbot.multilingual_handler.supported_languages,
            "default_language": "en",
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting supported languages: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/conversation/<user_id>', methods=['GET'])
def get_conversation(user_id):
    """Get conversation history for a specific user using persistent memory"""
    try:
        memory = get_memory_manager()
        history = memory.get_conversation_history(user_id)
        context = memory.get_conversation_context(user_id)

        return jsonify({
            "user_id": user_id,
            "history": history,
            "message_count": len(history),
            "has_context": bool(context),
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting conversation for user {user_id}: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/conversation/<user_id>', methods=['DELETE'])
def clear_conversation(user_id):
    """Clear conversation history for a specific user using persistent memory"""
    try:
        memory = get_memory_manager()
        cleared = memory.clear_conversation(user_id)

        if cleared:
            message = f"Conversation cleared for user {user_id}"
        else:
            message = f"No conversation found for user {user_id}"

        return jsonify({
            "message": message,
            "user_id": user_id,
            "cleared": cleared,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error clearing conversation for user {user_id}: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get multilingual chatbot statistics"""
    try:
        chatbot = get_chatbot()
        memory = get_memory_manager()

        return jsonify({
            "status": "active",
            "chunks_count": len(chatbot.vector_db.chunks),
            "active_conversations": memory.get_active_users_count(),
            "supported_languages": list(chatbot.multilingual_handler.supported_languages.keys()),
            "model_info": {
                "llm_model": chatbot.llm_model,
                "embedding_model": chatbot.embedding_model,
                "api_version": chatbot.api_version
            },
            "cache_info": {
                "cache_size": len(chatbot.prompt_cache.cache),
                "cache_ttl_hours": chatbot.prompt_cache.ttl_hours
            },
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }), 500


if __name__ == '__main__':
    # Detect if running on Render or locally
    is_production = os.getenv('RENDER') is not None
    debug = not is_production

    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))

    logger.info(f"Environment: {'Production (Render)' if is_production else 'Development'}")
    logger.info(f"Starting Multilingual Flask server on {host}:{port} (debug={debug})")

    # Initialize components on startup
    try:
        get_chatbot()
        get_memory_manager()
        logger.info("All components pre-loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        exit(1)

    # Run the Flask app
    app.run(host=host, port=port, debug=debug)
