from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from ..core.agent import Agent
from ..core.builder import Parser, Preprocessor, Indexer
from ..core.reasoner import Retriever

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    parameters: Dict[str, Any] = {}

@router.post("/process")
async def process_query(request: QueryRequest):
    try:
        # Initialize components
        agent = Agent()
        parser = Parser()
        preprocessor = Preprocessor()
        indexer = Indexer()
        retriever = Retriever()

        # Agent handles the orchestration
        parsed_data = agent.handle_parsing(request.query, parser)
        processed_data = agent.handle_preprocessing(parsed_data, preprocessor)
        indexed_data = agent.handle_indexing(processed_data, indexer)
        result = agent.handle_retrieval(indexed_data, retriever)

        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
