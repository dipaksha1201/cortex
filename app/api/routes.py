from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict, Any
from ..core.builder.index import Indexer
import logging

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/index")
async def index_file(
    project_name: str = Form(...),  # Accept project_name from form-data
    file: UploadFile = File(...)
):
    """
    Index a file for a given project
    
    Args:
        project_name: Name of the project (required)
        file: The file to be indexed
    """
    logger.info(f"Received indexing request for project: {project_name}, file: {file.filename}")
    
    try:
        if not project_name.strip():
            logger.warning("Empty project name provided")
            raise HTTPException(
                status_code=400,
                detail="project_name is required"
            )
            
        # Initialize indexer
        logger.debug("Initializing indexer")
        indexer = Indexer()
        
        # Index the file
        logger.info(f"Starting indexing process for {file.filename}")
        success = indexer.index(file, project_name)
        
        if success:
            logger.info(f"Successfully indexed file {file.filename} for project {project_name}")
            return {
                "status": "success",
                "message": f"File {file.filename} indexed successfully for project {project_name}"
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

