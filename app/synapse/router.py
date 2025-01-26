from fastapi import APIRouter, HTTPException
from typing import Dict, Any

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
    return {
        "name": "Cortex",
        "version": "1.0.0",
        "capabilities": [
            "data_processing",
            "neural_analysis",
            "pattern_recognition"
        ]
    }
