from pymongo import MongoClient
from datetime import datetime
import os

class MongoMessageDB:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI"))
        self.db = self.client["tutor_app"]
        self.collection = self.db["messages"]

    def add_message(self, chat_id, user_id, role, content):
        """Store a message inside MongoDB."""
        self.collection.insert_one({
            "chatId": chat_id,
            "userId": user_id,
            "role": role,      # "user" or "assistant"
            "content": content,
            "timestamp": datetime.utcnow()
        })

    def get_last_messages(self, chat_id, limit=6):
        """Return last N messages in descending order."""
        cursor = (
            self.collection
                .find({"chatId": chat_id})
                .sort("timestamp", -1)
                .limit(limit)
        )
        messages = list(cursor)
        return messages[::-1]  # return in oldest → newest order
