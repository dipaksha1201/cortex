from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from app.data_layer.services import DocumentService
from app.services.doc import Documents
from ..core.builder.index import Indexer
import logging

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/index")
async def index_file(
    user_name: str,
    file: UploadFile = File(...)
):
    """
    Index a file for a given project
    
    Args:
        project_name: Name of the project (required)
        file: The file to be indexed
    """
    logger.info(f"Received indexing request for user: {user_name}, file: {file.filename}")
     
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

class DocumentColumnRequest(BaseModel):
    user_id: str
    file_name: str
    column_name: str

@router.post("/document-column")
async def get_document_column(request: DocumentColumnRequest):
    """Get a specific column from a document"""
    try:
        service = Documents()
        column_value = service.get_document_column_info(request.user_id, request.file_name, request.column_name)
        return jsonable_encoder(column_value)
    except Exception as e:
        logger.error(f"Error retrieving document column: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document column: {str(e)}") 