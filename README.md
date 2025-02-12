# Cortex API

A powerful neural engine for processing and analyzing data built with FastAPI.

## Project Structure

```
cortex/
├── app/
│   ├── main.py           # Main FastAPI application
│   ├── api/              # API routes and controllers
│   │   ├── __init__.py
│   │   ├── chat_api.py   # Chat API endpoints
│   │   └── routes.py     # General API endpoints
│   ├── config.py         # Configuration management
│   ├── core/             # Core system components
│   │   ├── __init__.py
│   │   ├── agent.py      # Agent for handling various tasks
│   │   ├── builder/      # Indexing and parsing components
│   │   ├── common/       # Common utilities
│   │   ├── interface/    # Interface definitions
│   │   ├── reasoner/     # Reasoning engine components
│   │   └── tools/        # Tools for various operations
│   ├── cortex/           # Cortex-specific components
│   │   ├── __init__.py
│   │   ├── _constants.py # Constants used in Cortex
│   │   ├── _schemas.py   # Schemas for data validation
│   │   ├── _settings.py  # Settings for Cortex
│   │   ├── _utils.py     # Utility functions
│   │   ├── brain.py      # Main brain logic
│   │   ├── memory_functions.py # Memory-related functions
│   │   ├── observer.py   # Observer for monitoring
│   │   └── tools.py      # Tools specific to Cortex
│   ├── data_layer/       # Data access layer
│   │   ├── __init__.py
│   │   ├── db_config.py  # Database configuration
│   │   ├── models/       # Data models
│   │   ├── services/     # Services for data operations
│   │   └── sqlite_config.py # SQLite configuration
│   ├── initialization.py # Initialization logic
│   ├── logging_config.py # Logging configuration
│   ├── services/         # Additional services
│   │   ├── __init__.py
│   │   ├── chat.py       # Chat service
│   └── storage/          # Storage-related components
│       ├── __init__.py
│       ├── disk_store.py # Disk storage management
│       └── pinecone.py   # Pinecone storage management
│   ├── synapse/          # System-related endpoints
│       ├── __init__.py
│       └── router.py     # System API endpoints
├── checkpoints.sqlite    # SQLite database file
├── logs/                 # Log files
│   ├── agent.log
│   ├── app.log
│   ├── indexing.log
│   ├── memory.log
│   └── reasoning.log
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the server with:
```bash
uvicorn app.main:app --loop asyncio 
```

The API will be available at http://localhost:8000

## API Documentation

Once the server is running, you can access:
- Interactive API documentation: http://localhost:8000/docs
- Alternative documentation: http://localhost:8000/redoc

## Components

- **api**: API routes and controllers
- **config.py**: Configuration management
- **core**: Core system components
- **cortex**: Cortex-specific components
- **data_layer**: Data access layer
- **initialization.py**: Initialization logic
- **logging_config.py**: Logging configuration
- **services**: Additional services
- **storage**: Storage-related components
- **synapse**: System-related endpoints

## Reasoning Example

```json
{
    "reasoning": [
        {
            "query": "example_query",
            "properties": "example_properties",
            "context": "example_context"
        },
        {
            "query": "another_query",
            "properties": "another_properties",
            "context": "another_context"
        }
    ],
    "final_answer": "example_final_answer"
}
