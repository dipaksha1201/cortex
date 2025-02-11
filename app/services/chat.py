import logging
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from app.cortex._schemas import GraphConfig
from app.data_layer.services.conversation_service import ConversationService
from ..core.reasoner.resoning_engine import ReasoningEngine
from app.data_layer.models.conversation import Message
from app.cortex.brain import cortex
from app.cortex.observer import observer
from app.logging_config import agent_logger
from app.logging_config import memory_logger

class Chat: 
    
    @staticmethod
    async def process_chat_message(payload: dict):
        # Create the Message object from the incoming JSON payload
        message = Message(
            sender=payload["sender"],
            type=payload["type"],
            content=payload["content"]
        )
        # Instantiate the conversation service
        conversation_service = ConversationService()
        # Process user message
        conversation = conversation_service.store_message(
            message=message,
            user_id=payload["user_id"],
            conversation_id=payload.get("conversation_id")
        )
        
        config = {
                "configurable": GraphConfig(
                    user_id=payload["user_id"],
                    thread_id=conversation.id,
                ),
            }
        
        cortex.invoke(
            {
                "messages": ("human", payload["content"]),
            },
            config,
            stream_mode="values",
        )
        cortex_state = cortex.get_state(config)
        state = cortex_state.values
        # state['messages'] = state['messages'][:4]
        # state['messages'].append(("human", payload["content"]))
        # memory_logger.info(f"Cortex state messages:\n {state['messages']}")
        # state.pop("thread_id", None) 
        # response = await observer.ainvoke({
        #         "messages": state["messages"],
        #     }, {
        #      "configurable": GraphConfig(
        #             user_id=payload["user_id"],
        #             thread_id=payload["conversation_id"],
        #         ),},
        #      stream_mode="values", 
        # )
        
        # Run reasoning engine and process reasoning message
        # agent_logger.info("response: %s", response)
        id = str(conversation.id)
        return state["output"] , id

    @staticmethod
    def process_chat_payload(payload: dict):
        try:
            return Chat.process_chat_message(payload)
        except Exception as e:
            logging.error(f"Error processing chat payload: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    def process_get_all_conversations(user_id: str):
        service = ConversationService()
        conversations = service.get_user_conversations(user_id=user_id)
        return jsonable_encoder(conversations)
