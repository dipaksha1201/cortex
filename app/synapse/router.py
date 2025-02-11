from fastapi import APIRouter
from typing import Dict, Any

from app.data_layer.db_config import MongoDBConfig

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify system status
    """
    return {
        "status": "healthy",
        "components": {
            "api": "operational",
            "database": "connected"
        }
    }

@router.get("/info")
async def system_info() -> Dict[str, Any]:
    """
    Get system information and capabilities
    """
    mongo = MongoDBConfig()
    client = mongo.connect()
    collections = client.list_collection_names()
    return {
        "name": "Cortex",
        "version": "1.0.0",
        "collections": collections,
        "capabilities": [
            "data_processing",
            "neural_analysis",
            "pattern_recognition"
        ]
    }
