#!/usr/bin/env python3
"""
Simple script to run the Flask RAG API server
"""
import os
import sys
from flask_app import app, get_chatbot, logger

def main():
    """Main function to start the Flask server"""
    print("🚀 Starting RAG Chatbot Flask API...")
    print("=" * 50)

    # Configuration
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'  # Default to True for development

    print(f"📡 Server configuration:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")

    # Pre-initialize chatbot
    try:
        print("\n🤖 Initializing RAG Chatbot...")
        chatbot = get_chatbot()
        print(f"✅ RAG Chatbot ready with {len(chatbot.vector_db.chunks)} chunks")

        print(f"\n🌐 API Endpoints available:")
        print(f"   Health Check: http://{host}:{port}/")
        print(f"   Chat (Main):  http://{host}:{port}/chat")
        print(f"   Chat Simple:  http://{host}:{port}/chat/simple")
        print(f"   Chat Detail:  http://{host}:{port}/chat/detailed")
        print(f"   Stats:        http://{host}:{port}/stats")

        print(f"\n🔗 For webhook integration, use:")
        print(f"   PYTHON_API_URL=http://{host}:{port}/chat")

        print(f"\n🚀 Starting server...")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Failed to initialize RAG Chatbot: {e}")
        sys.exit(1)

    # Start Flask server
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    main()
