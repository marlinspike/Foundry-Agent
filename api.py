import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional, Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, APIRouter, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import HTTPConnection
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables before importing orchestrator
load_dotenv(override=True)

from orchestrator import build_orchestrator, OrchestratorWorkflow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender (e.g., 'user', 'assistant')")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    user_text: str = Field(..., description="The user's input text")
    history: Optional[List[Message]] = Field(default=None, description="Optional chat history")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The assistant's response")

class StreamEvent(BaseModel):
    type: str = Field(..., description="The type of the event (e.g., 'status', 'answer', 'error', 'done')")
    text: str = Field(..., description="The content of the event")

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def get_orchestrator(conn: HTTPConnection) -> OrchestratorWorkflow:
    """Dependency to inject the orchestrator instance."""
    return conn.app.state.orchestrator

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

chat_router = APIRouter(tags=["Chat"])
health_router = APIRouter(tags=["System"])

@health_router.get("/health")
async def health_check():
    """
    Simple health check endpoint to verify the API is running.
    """
    return {"status": "ok", "service": "Foundry-Agent API"}

@chat_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, 
    orch: OrchestratorWorkflow = Depends(get_orchestrator)
):
    """
    Standard REST endpoint for chat.
    """
    try:
        history_dicts = [msg.model_dump() for msg in request.history] if request.history else None
        answer = await orch.run(user_text=request.user_text, history=history_dicts)
        return ChatResponse(answer=answer)
    except Exception as e:
        logger.exception("Error processing chat request")
        raise HTTPException(status_code=500, detail=str(e))

@chat_router.websocket("/ws/chat")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    orch: OrchestratorWorkflow = Depends(get_orchestrator)
):
    """
    WebSocket endpoint for streaming chat responses.
    Expects JSON messages matching the ChatRequest schema.
    Sends back JSON messages with event types: 'status', 'answer', 'error', 'done'.
    """
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            await handle_websocket_message(websocket, data, orch)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("Unexpected WebSocket error")

async def handle_websocket_message(websocket: WebSocket, data: Dict[str, Any], orch: OrchestratorWorkflow):
    """Handles a single WebSocket message."""
    try:
        request = ChatRequest(**data)
    except Exception as e:
        await websocket.send_json(StreamEvent(type="error", text=f"Invalid request format: {e}").model_dump())
        return
        
    history_dicts = [msg.model_dump() for msg in request.history] if request.history else None
    
    try:
        async for event_type, text in orch.run_stream(user_text=request.user_text, history=history_dicts):
            await websocket.send_json(StreamEvent(type=event_type, text=text).model_dump())
        await websocket.send_json(StreamEvent(type="done", text="").model_dump())
    except Exception as e:
        logger.exception("Error during stream")
        await websocket.send_json(StreamEvent(type="error", text=f"Internal server error: {e}").model_dump())

# ---------------------------------------------------------------------------
# Lifespan & App Setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the lifecycle of the OrchestratorWorkflow.
    Initializes the orchestrator on startup and cleans it up on shutdown.
    """
    logger.info("Starting up Orchestrator...")
    async with build_orchestrator() as orch:
        app.state.orchestrator = orch
        logger.info("Orchestrator started successfully.")
        yield
    logger.info("Orchestrator shut down.")

def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    app = FastAPI(
        title="Foundry-Agent API",
        description="REST API and WebSocket interface for the Multi-Agent Orchestrator",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health_router)
    app.include_router(chat_router)
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
