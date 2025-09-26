#!/usr/bin/env python3
"""
Simple script to run the Multilingual Flask RAG API server
"""
import os
import sys
from flask_app import app, get_chatbot, logger

def main():
    """Main function to start the Flask server"""
    print("🚀 Starting Multilingual RAG Chatbot Flask API...")
    print("=" * 60)

    # Configuration
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = True  # Set to False for production deployment

    print(f"📡 Server configuration:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")

    # Pre-initialize multilingual chatbot
    try:
        print("\n🤖 Initializing Multilingual RAG Chatbot...")
        chatbot = get_chatbot()
        print(f"✅ Multilingual RAG Chatbot ready with {len(chatbot.vector_db.chunks)} chunks")
        print(f"🌍 Supported languages: {list(chatbot.multilingual_handler.supported_languages.values())}")

        print(f"\n🌐 API Endpoints available:")
        print(f"   Health Check:     http://{host}:{port}/")
        print(f"   Chat (Main):      http://{host}:{port}/chat")
        print(f"   Chat Simple:      http://{host}:{port}/chat/simple")
        print(f"   Chat Detailed:    http://{host}:{port}/chat/detailed")
        print(f"   Languages:        http://{host}:{port}/languages")  # NEW ENDPOINT
        print(f"   Stats:            http://{host}:{port}/stats")

        print(f"\n🔗 For webhook integration, use:")
        print(f"   PYTHON_API_URL=http://{host}:{port}/chat")

        print(f"\n📝 Test multilingual queries:")
        print(f"   English:  'What are your business hours?'")
        print(f"   Filipino: 'Anong oras kayo bukas?'")
        print(f"   Cebuano:  'Asa dapit nahimutang inyong branch?'")

        print(f"\n🚀 Starting server...")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Failed to initialize Multilingual RAG Chatbot: {e}")
        sys.exit(1)

    # Start Flask server
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    main()
