from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

# Configure logging
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/app.log"),  # Log file path
                        logging.StreamHandler()            # Console output
                    ])
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cortex API",
    description="A powerful neural engine for processing and analyzing data",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {
        "message": "Welcome to Cortex API",
        "status": "operational"
    }

if __name__ == "__main__":
    logger.info("Starting the application...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
