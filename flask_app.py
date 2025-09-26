from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from typing import Dict, List, Any
import logging
from datetime import datetime

# Import your RAG chatbot
from rag_app import RAGChatbot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global chatbot instance (singleton pattern)
chatbot_instance = None

def get_chatbot():
    """Get or create the RAG chatbot instance"""
    global chatbot_instance
    if chatbot_instance is None:
        logger.info("Initializing RAG Chatbot...")
        chatbot_instance = RAGChatbot()
        if not chatbot_instance.is_initialized:
            logger.error("Failed to initialize RAG Chatbot")
            raise RuntimeError("RAG Chatbot initialization failed")
        logger.info(f"RAG Chatbot initialized with {len(chatbot_instance.vector_db.chunks)} chunks")
    return chatbot_instance

# In-memory conversation storage (for development)
# In production, you might want to use Redis or a database
conversation_memories = {}

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        chatbot = get_chatbot()
        return jsonify({
            "status": "healthy",
            "message": "RAG Chatbot API is running",
            "chunks_count": len(chatbot.vector_db.chunks),
            "initialized": chatbot.is_initialized,
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
    """Main chat endpoint for webhook integration"""
    try:
        # Get request data
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        message = data.get('message', '').strip()
        history = data.get('history', [])
        user_id = data.get('user_id', 'default')  # Optional user ID for conversation tracking

        if not message:
            return jsonify({"error": "No message provided"}), 400

        logger.info(f"Received message from user {user_id}: {message}")

        # Get chatbot instance
        chatbot = get_chatbot()

        # Generate response
        result = chatbot.generate_response(message)

        # Update conversation history
        updated_history = history.copy()
        updated_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        updated_history.append({
            "role": "assistant",
            "content": result["response"],
            "timestamp": datetime.now().isoformat(),
            "confidence": result["confidence"],
            "sources_count": len(result["sources"])
        })

        # Keep only last 20 messages to prevent memory issues
        if len(updated_history) > 20:
            updated_history = updated_history[-20:]

        # Store conversation in memory (optional)
        conversation_memories[user_id] = updated_history

        logger.info(f"Generated response for user {user_id} with confidence {result['confidence']:.2f}")

        # Return response in format expected by webhook
        return jsonify({
            "response": result["response"],
            "history": updated_history,
            "confidence": result["confidence"],
            "sources_count": len(result["sources"]),
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
    """Simplified chat endpoint that only returns the response text"""
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

        # Return just the response text
        return jsonify({
            "response": result["response"]
        }), 200

    except Exception as e:
        logger.error(f"Error in simple chat endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/chat/detailed', methods=['POST'])
def chat_detailed():
    """Detailed chat endpoint with full RAG information"""
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

        # Return detailed information
        return jsonify({
            "response": result["response"],
            "confidence": result["confidence"],
            "sources": result["sources"],
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "llm_model": chatbot.llm_model,
                "embedding_model": chatbot.embedding_model,
                "chunks_total": len(chatbot.vector_db.chunks)
            }
        }), 200

    except Exception as e:
        logger.error(f"Error in detailed chat endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/conversation/<user_id>', methods=['GET'])
def get_conversation(user_id):
    """Get conversation history for a specific user"""
    try:
        history = conversation_memories.get(user_id, [])
        return jsonify({
            "user_id": user_id,
            "history": history,
            "message_count": len(history),
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
    """Clear conversation history for a specific user"""
    try:
        if user_id in conversation_memories:
            del conversation_memories[user_id]
            message = f"Conversation cleared for user {user_id}"
        else:
            message = f"No conversation found for user {user_id}"

        return jsonify({
            "message": message,
            "user_id": user_id,
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
    """Get chatbot statistics"""
    try:
        chatbot = get_chatbot()

        return jsonify({
            "status": "active",
            "chunks_count": len(chatbot.vector_db.chunks),
            "active_conversations": len(conversation_memories),
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
    # Get configuration from environment variables
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = False  # Set to False for production deployment

    logger.info(f"Starting Flask server on {host}:{port} (debug={debug})")

    # Initialize chatbot on startup
    try:
        get_chatbot()
        logger.info("RAG Chatbot pre-loaded successfully")
    except Exception as e:
        logger.error(f"Failed to pre-load RAG Chatbot: {e}")
        exit(1)

    # Run the Flask app
    app.run(host=host, port=port, debug=debug)
