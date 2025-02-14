from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any
from fastapi.responses import StreamingResponse

from app.data_layer.services import MemoryService
from ..core.reasoner.resoning_engine import ReasoningEngine
import logging
from fastapi.encoders import jsonable_encoder
from app.core.reasoner.retrievers.sparse_retriever import SparseRetriever

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/memories/all")
async def get_all_memories(user_id: str):
    try:
        logger.info("Retrieving all memories")
        service = MemoryService()
        documents = service.get_user_memories(user_id=user_id)
        return jsonable_encoder(documents)
    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving memories: {str(e)}"
        )

class ChatRequest(BaseModel):
    username: str
    query: str

@router.post("/reason")
async def retrieve_file(
    request: ChatRequest = Body(...),
):
    logger.info(f"Received chat request")
    try:
        username = request.username
        query = request.query
        logger.debug(f"Processing query for project '{username}': {query}")
        
        # Add your processing logic here
        reasoning = ReasoningEngine(username=username, query=query)
        response = reasoning.start_reasoning()
        
        return jsonable_encoder(response)
    except Exception as e:
        logger.error(f"Error processing reasoning api request: {e}")
        return {"error": str(e)}

class SparseRetrievalRequest(BaseModel):
    query: str
    index_name: str
    score_threshold: float = 0.45

class SparseRetrievalResponse(BaseModel):
    results: list
    combined_text: str

@router.post("/sparse-retrieve")
async def test_sparse_retriever(request: SparseRetrievalRequest):
    """Test endpoint for sparse retriever"""
    try:
        retriever = SparseRetriever(request.index_name)
        nodes, text = retriever.retrieve(
            request.query, 
            score_threshold=request.score_threshold
        )
        return {
            "results": [n.dict() for n in nodes],
            "combined_text": text
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sparse retrieval failed: {str(e)}"
        )