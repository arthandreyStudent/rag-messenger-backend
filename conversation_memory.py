import redis
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    def __init__(self):
        self.memory_fallback = {} # Always initialize fallback memory
        """Initialize Redis-based conversation memory manager"""
        try:
            # Connect to Redis (Render provides REDIS_URL automatically)
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)

            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis: {redis_url}")

        except Exception as e:
            logger.warning(f"⚠️  Redis connection failed: {e}")
            logger.info("🔄 Falling back to in-memory storage for development")
            self.redis_client = None

        # Memory settings
        self.max_history_length = 20  # Keep last 20 exchanges
        self.memory_ttl = 86400 * 7  # 7 days expiry

    def _is_redis_available(self) -> bool:
        """Check if Redis is available"""
        return self.redis_client is not None

    def get_conversation_key(self, user_id: str) -> str:
        """Generate Redis key for user conversation"""
        return f"conversation:{user_id}"

    def get_context_key(self, user_id: str) -> str:
        """Generate Redis key for conversation context"""
        return f"context:{user_id}"

    def add_message(self, user_id: str, role: str, content: str,
                    language: str = "en", metadata: Dict = None) -> None:
        """Add message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "language": language,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        # Get existing conversation
        history = self.get_conversation_history(user_id)
        history.append(message)

        # Keep only recent messages
        if len(history) > self.max_history_length:
            history = history[-self.max_history_length:]

        # Store conversation
        if self._is_redis_available():
            try:
                conversation_key = self.get_conversation_key(user_id)
                self.redis_client.setex(
                    conversation_key,
                    self.memory_ttl,
                    json.dumps(history)
                )
                logger.debug(f"Stored message for user {user_id} in Redis")
            except Exception as e:
                logger.error(f"Redis store error: {e}, falling back to memory")
                self.memory_fallback[user_id] = history
        else:
            # Fallback to in-memory storage
            self.memory_fallback[user_id] = history

    def get_conversation_history(self, user_id: str) -> List[Dict]:
        """Get conversation history for user"""
        if self._is_redis_available():
            try:
                conversation_key = self.get_conversation_key(user_id)
                history_json = self.redis_client.get(conversation_key)

                if history_json:
                    return json.loads(history_json)
            except Exception as e:
                logger.error(f"Redis get error: {e}, checking fallback")

        # Check fallback memory
        return self.memory_fallback.get(user_id, [])

    def get_conversation_context(self, user_id: str) -> str:
        """Build conversation context for LLM"""
        history = self.get_conversation_history(user_id)

        if not history:
            return ""

        # Build context from recent exchanges
        context_parts = []
        for msg in history[-10:]:  # Last 10 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")

        return "\n".join(context_parts)

    def update_conversation_summary(self, user_id: str, summary: str) -> None:
        """Store conversation summary for long-term context"""
        context_data = {
            "summary": summary,
            "last_updated": datetime.now().isoformat(),
            "user_id": user_id
        }

        if self._is_redis_available():
            try:
                context_key = self.get_context_key(user_id)
                self.redis_client.setex(
                    context_key,
                    self.memory_ttl * 2,  # Keep summaries longer
                    json.dumps(context_data)
                )
            except Exception as e:
                logger.error(f"Redis summary store error: {e}")

    def get_conversation_summary(self, user_id: str) -> Optional[str]:
        """Get conversation summary for context"""
        if self._is_redis_available():
            try:
                context_key = self.get_context_key(user_id)
                context_json = self.redis_client.get(context_key)

                if context_json:
                    context_data = json.loads(context_json)
                    return context_data.get("summary")
            except Exception as e:
                logger.error(f"Redis summary get error: {e}")

        return None

    def clear_conversation(self, user_id: str) -> bool:
        """Clear conversation history for user"""
        cleared = False

        if self._is_redis_available():
            try:
                conversation_key = self.get_conversation_key(user_id)
                context_key = self.get_context_key(user_id)

                deleted_conv = self.redis_client.delete(conversation_key)
                deleted_context = self.redis_client.delete(context_key)

                cleared = bool(deleted_conv or deleted_context)
            except Exception as e:
                logger.error(f"Redis clear error: {e}")

        # Also clear from fallback
        if user_id in self.memory_fallback:
            del self.memory_fallback[user_id]
            cleared = True

        return cleared

    def get_active_users_count(self) -> int:
        """Get count of users with active conversations"""
        if self._is_redis_available():
            try:
                keys = self.redis_client.keys("conversation:*")
                return len(keys)
            except Exception as e:
                logger.error(f"Redis count error: {e}")

        return len(self.memory_fallback)

    def health_check(self) -> Dict[str, Any]:
        """Check Redis health and return status"""
        status = {
            "redis_available": False,
            "fallback_users": len(self.memory_fallback),
            "connection_info": "In-memory fallback"
        }

        if self._is_redis_available():
            try:
                self.redis_client.ping()
                status.update({
                    "redis_available": True,
                    "active_conversations": self.get_active_users_count(),
                    "connection_info": str(self.redis_client.connection_pool.connection_kwargs.get('host', 'unknown'))
                })
            except Exception as e:
                status["connection_error"] = str(e)

        return status
