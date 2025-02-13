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
    try:
        output, id = await Chat.process_chat_payload(payload)
        return {"conversation_id": id, "response": output}
    except Exception as e:
        logger.error("Error in /respond endpoint: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")

@chat_router.get("/get/conversation")
async def get_conversation_api(conversation_id: str):
    try:
        service = ConversationService()
        conversation = service.get_conversation(conversation_id=conversation_id)
        if conversation:
            return {"conversation": jsonable_encoder(conversation)}
        else:
            # Return a clear 404 if the conversation is not found
            raise HTTPException(
                status_code=404,
                detail=f"Conversation with id '{conversation_id}' not found"
            )
    except Exception as e:
        logger.error("Error in /get/conversation endpoint: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")

@chat_router.get("/get/conversation/all")
async def get_all_conversations_api(user_id: str):
    try:
        conversations = Chat.process_get_all_conversations(user_id)
        return {"conversations": conversations}
    except Exception as e:
        logger.error("Error in /get/conversation/all endpoint: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")