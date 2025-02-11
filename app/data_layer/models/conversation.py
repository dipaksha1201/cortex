from pydantic import BaseModel, Field, field_serializer
from datetime import datetime
from typing import Dict, Literal, List, Optional, Union
import uuid
from bson import ObjectId
        
class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Auto-generate UUID
    sender: str
    type: Literal["query", "internal_knowledge", "memory", "from_conversation"]  # Extendable
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)  # Auto-generate current timestamp

    class Config:
        extra = "allow"  # Allows additional fields

class Conversation(BaseModel):
    id: Optional[ObjectId] = Field(default=None, alias="_id")
    user_id: str = Field(..., description="Identifier for the user")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the conversation was created")
    last_updated: datetime = Field(default_factory=datetime.utcnow,
                                   description="Timestamp when the conversation was last updated")
    title: Optional[str] = Field(default=None, description="Optional title or subject of the conversation")
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    output_table: List[Dict] = Field(default_factory=list, description="List of dictionaries representing output table")
    summary: Optional[str] = Field(default=None, description="Optional conversation summary")
    highlight: Optional[str] = Field(default=None, description="Optional conversation highlight")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata about the conversation")

    @field_serializer("id")
    def serialize_objectid(self, v: ObjectId) -> str:
        return str(v)

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
