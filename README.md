# Cortex API

A powerful neural engine for processing and analyzing data built with FastAPI.

## Project Structure

```
cortex/
├── app/
│   ├── main.py           # Main FastAPI application
│   ├── nucleus/          # Core system components
│   │   ├── __init__.py
│   │   └── config.py     # Configuration management
│   └── synapse/          # API routes and controllers
│       ├── __init__.py
│       └── router.py     # API endpoints
├── requirements.txt      # Project dependencies
└── README.md            # This file
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

- **nucleus**: Core system components and configuration
- **synapse**: API routes and controllers

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
