from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body
from pydantic import BaseModel
from typing import Dict, Any

from app.data_layer.services.document_service import DocumentService
from ..core.builder.index import Indexer
from ..core.reasoner.resoning_engine import ReasoningEngine
import logging
from fastapi.encoders import jsonable_encoder

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/index")
async def index_file(
    user_name: str,  # Accept project_name from form-data
    file: UploadFile = File(...)
):
    """
    Index a file for a given project
    
    Args:
        project_name: Name of the project (required)
        file: The file to be indexed
    """
    logger.info(f"Received indexing request for user: {user_name}, file: {file.filename}")
    
    try:
        if not user_name.strip():
            logger.warning("Empty user name provided")
            raise HTTPException(
                status_code=400,
                detail="user_name is required"
            )
            
        # Initialize indexer
        logger.debug("Initializing indexer")
        indexer = Indexer()
        
        # Index the file
        logger.info(f"Starting indexing process for {file.filename}")
        success = await indexer.index(file, user_name)
        
        if success:
            logger.info(f"Successfully indexed file {file.filename} for user {user_name}")
            return {
                "status": "success",
                "message": f"File {file.filename} indexed successfully for user {user_name}"
            }
        else:
            logger.error(f"Failed to index file {file.filename}")
            raise HTTPException(
                status_code=500,
                detail="Failed to index file"
            )
            
    except Exception as e:
        logger.error(f"Error indexing file {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error indexing file: {str(e)}"
        )

@router.get("/documents/all")
async def get_all_documents(user_id: str):
    try:
        logger.info("Retrieving all documents")
        service = DocumentService()
        documents = service.get_user_documents(user_id=user_id)
        return jsonable_encoder(documents)
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving documents: {str(e)}"
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