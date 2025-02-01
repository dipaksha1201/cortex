from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import api_router
from .synapse import synapse_router
from .logging_config import reasoning_logger
import logging
import uvicorn
import os

# Get absolute path to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(project_root, "logs")
log_file_path = os.path.join(log_dir, "app.log")

# Create logs directory
os.makedirs(log_dir, exist_ok=True)

print(f"Log directory: {log_dir}")  # Debug print
print(f"Log file path: {log_file_path}")  # Debug print

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),  # Append mode
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Export reasoning_logger for use in other modules
__all__ = ["reasoning_logger"]

app = FastAPI(
    title="Cortex API",
    description="AI-powered document processing and analysis API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(api_router, prefix="/api", tags=["API"])
app.include_router(synapse_router, prefix="/synapse", tags=["System"])

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    reasoning_logger.info("Root endpoint called with reasoning logger")
    return {
        "message": "Welcome to Cortex API",
        "status": "operational"
    }

if __name__ == "__main__":
    logger.info("Starting the application...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
