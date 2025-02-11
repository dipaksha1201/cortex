from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from app.data_layer.services.conversation_service import ConversationService
import logging
from app.services.chat import Chat

chat_router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Removed: process_chat_payload and process_get_all_conversations helper functions

@chat_router.post("/respond")
async def respond_api(payload: dict = Body(...)):
    output , id = await Chat.process_chat_payload(payload)
    return {"conversation_id": id, "response": output}

@chat_router.get("/get/conversation")
async def get_conversation_api(conversation_id: str):
    service = ConversationService()
    conversation = service.get_conversation(conversation_id=conversation_id)
    if conversation:
        return {"conversation": jsonable_encoder(conversation)}
    else:
        return {"conversation": f"Conversation with id '{conversation_id}' not found"}

@chat_router.get("/get/conversation/all")
async def get_all_conversations_api(user_id: str):
    conversations = Chat.process_get_all_conversations(user_id)
    return {"conversations": conversations}