from bson import ObjectId
from pydantic import BaseModel, Field, field_serializer
from typing import List

class Memory(BaseModel):
    id: ObjectId = Field(None, alias="_id")
    conversation_id: str
    user_id: str
    summary: str
    title: str = ""
    highlights: str = ""
    last_update_count: int = 0
    
    @field_serializer("id")
    def serialize_objectid(self, v: ObjectId) -> str:
        return str(v)

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        }