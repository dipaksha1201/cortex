from typing import Dict, List, Optional
from datetime import datetime
from pymongo import ReturnDocument
from app.data_layer.db_config import MongoDBConfig
from bson import ObjectId
from app.data_layer.models.document import Document
from app.data_layer.models.memory import Memory
from app.logging_config import memory_logger as logger

class MemoryService:
    def __init__(self):
        # In-memory store for conversations; replace with a persistent store in production.
        db_config = MongoDBConfig()
        self.db = db_config.connect()
        logger.info("MemoryService initialized and database connected.")

    def insert_memory(self, memory: Memory, is_new:bool) -> Memory:
        if is_new:
            # Insert into MongoDB 'documents' collection
            result = self.db["memories"].insert_one(memory.model_dump())
            # Assign the generated _id to the memory dict
            memory.id = str(result.inserted_id)
            
        else:
            # Update the existing memory in MongoDB 'memories' collection
            result = self.db["memories"].find_one_and_update(
                {"_id": ObjectId(memory.id)},
                {"$set": memory.model_dump()},
                return_document=ReturnDocument.AFTER
            )
            memory = Memory(**result)
        # Return a new memory instance with the inserted data
        return memory
    
    def get_memory_for_conversation(self, conversation_id: str) -> Memory:
        """Retrieve a memory by its memory id."""
        logger.debug("Retrieving memory with id: %s", conversation_id)
        
        # ensure that you use the correct one for your MongoDB configuration.
        memory = self.db["memories"].find_one({"conversation_id": conversation_id})
        if memory is None:
            logger.error("Memory with thread id '%s' not found", conversation_id)
            return False
            # raise ValueError(f"Memory with id '{conversation_id}' not found.")
        logger.info("Memory with thread id '%s' retrieved successfully", conversation_id)
        return Memory(**memory)
    
    def get_user_memories(self, user_id: str) -> List[Document]:
        """Retrieve all memories for a specific user."""
        logger.info("Retrieving memories for user_id: %s", user_id)
        # Query the database for memories that match the given user_id.
        documents = self.db["memories"].find({"user_id": user_id})
        memories = [Memory(**doc) for doc in documents]
        logger.info("Found %d memories for user '%s'", len(memories), user_id)
        return memories