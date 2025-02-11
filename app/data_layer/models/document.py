from bson import ObjectId
from pydantic import BaseModel, Field, field_serializer
from typing import List

class Document(BaseModel):
    id: ObjectId = Field(None, alias="_id")
    user_id: str
    name: str
    type: str
    summary: str
    highlights: List[str]
    
    @field_serializer("id")
    def serialize_objectid(self, v: ObjectId) -> str:
        return str(v)

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        }