#!/usr/bin/env python3
"""
FastAPI Server for AI Legal Assistant
====================================

A simple and fast API server to host the AI Legal Assistant agent.
Provides REST endpoints for legal queries with CORS support for frontend integration.

Features:
- RESTful API endpoints
- Async support for better performance
- CORS enabled for frontend integration
- Error handling and logging
- Health check endpoint
- Simple chat interface

Usage:
    python api_server.py
    # Server will start on http://localhost:8000
    # API docs available at http://localhost:8000/docs
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "agent"))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global assistant instance
legal_assistant = None

# Pydantic models for request/response
class LegalQuery(BaseModel):
    question: str = Field(..., description="Legal question to ask", min_length=1)
    context: Optional[str] = Field(None, description="Additional context for the question")
    include_case_law: bool = Field(False, description="Include case law analysis")
    include_terms: bool = Field(True, description="Include legal terms analysis")

class LegalResponse(BaseModel):
    success: bool = Field(..., description="Whether the request was successful")
    response: Optional[str] = Field(None, description="The legal assistant's response")
    sources: Optional[List[str]] = Field(None, description="Sources used for the response")
    legal_terms: Optional[Dict[str, Any]] = Field(None, description="Legal terms found in the query")
    case_law: Optional[Dict[str, Any]] = Field(None, description="Case law analysis if requested")
    confidence: Optional[float] = Field(None, description="Confidence score of the response")
    error: Optional[str] = Field(None, description="Error message if failed")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    assistant_initialized: bool = Field(..., description="Whether the legal assistant is initialized")
    version: str = Field(..., description="API version")

# Initialize the legal assistant on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the legal assistant"""
    global legal_assistant
    
    logger.info("🚀 Starting AI Legal Assistant API Server...")
    
    try:
        # Initialize the legal assistant
        from agent.legal_assistant import SimpleLegalAssistant
        
        logger.info("📚 Initializing Legal Assistant...")
        legal_assistant = SimpleLegalAssistant()
        
        # Initialize in background
        if legal_assistant.initialize():
            logger.info("✅ Legal Assistant initialized successfully!")
        else:
            logger.error("❌ Failed to initialize Legal Assistant")
            legal_assistant = None
            
    except Exception as e:
        logger.error(f"❌ Failed to start Legal Assistant: {e}")
        legal_assistant = None
    
    yield
    
    # Cleanup on shutdown
    logger.info("🔄 Shutting down AI Legal Assistant API Server...")

# Create FastAPI app
app = FastAPI(
    title="AI Legal Assistant API",
    description="Free, open-source AI Legal Assistant for Indian laws",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Legal Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if legal_assistant and legal_assistant.is_initialized else "initializing",
        assistant_initialized=legal_assistant is not None and legal_assistant.is_initialized,
        version="1.0.0"
    )

@app.post("/ask", response_model=LegalResponse)
async def ask_legal_question(query: LegalQuery):
    """
    Ask a legal question to the AI assistant
    
    This endpoint processes legal questions and returns comprehensive answers
    with sources, legal terms analysis, and optional case law analysis.
    """
    
    if not legal_assistant:
        raise HTTPException(
            status_code=503, 
            detail="Legal Assistant is not initialized. Please try again later."
        )
    
    if not legal_assistant.is_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Legal Assistant is still initializing. Please try again in a few moments."
        )
    
    try:
        logger.info(f"📝 Processing legal query: {query.question[:100]}...")
          # Choose the appropriate method based on case law request
        if query.include_case_law and hasattr(legal_assistant, 'ask_question_with_case_law'):
            result = legal_assistant.ask_question_with_case_law(
                question=query.question,
                context=query.context,
                include_case_law=query.include_case_law
            )
        else:
            result = legal_assistant.ask_question(
                question=query.question,
                context=query.context,
                include_term_analysis=query.include_terms
            )
          # Format the response - convert legal_terms list to dict if needed
        legal_terms = result.get("legal_terms")
        if isinstance(legal_terms, list):
            # Convert list of legal terms to dictionary format
            legal_terms_dict = {}
            for term_info in legal_terms:
                if isinstance(term_info, dict) and 'term' in term_info and 'definition' in term_info:
                    legal_terms_dict[term_info['term']] = term_info['definition']
            legal_terms = legal_terms_dict
        
        response = LegalResponse(
            success=True,
            response=result.get("response", ""),
            sources=result.get("sources", []),
            legal_terms=legal_terms,
            case_law=result.get("case_law_analysis"),
            confidence=result.get("confidence")
        )
        
        logger.info("✅ Legal query processed successfully")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error processing legal query: {e}")
        return LegalResponse(
            success=False,
            error=str(e)
        )

@app.post("/chat", response_model=LegalResponse)
async def chat_with_assistant(query: LegalQuery):
    """
    Simple chat interface with the legal assistant
    
    This is an alias for the /ask endpoint for easier frontend integration.
    """
    return await ask_legal_question(query)

@app.get("/supported-acts")
async def get_supported_acts():
    """Get list of supported legal acts"""
    if not legal_assistant or not legal_assistant.is_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Legal Assistant is not initialized"
        )
    
    try:
        # Return the acts that are loaded
        acts_info = []
        if hasattr(legal_assistant.processor, 'loaded_acts'):
            acts_info = list(legal_assistant.processor.loaded_acts.keys())
        
        return {
            "supported_acts": acts_info,
            "total_acts": len(acts_info)
        }
    except Exception as e:
        logger.error(f"Error getting supported acts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    if not legal_assistant or not legal_assistant.is_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Legal Assistant is not initialized"
        )
    
    try:
        stats = {
            "status": "active",
            "initialized": legal_assistant.is_initialized,
            "indian_kanoon_available": legal_assistant.indian_kanoon_client is not None,
            "case_law_analyzer_available": hasattr(legal_assistant, 'case_law_analyzer'),
            "legal_terms_available": hasattr(legal_assistant, 'legal_terms_integrator')
        }
        
        # Add document count if available
        if hasattr(legal_assistant.rag_system, 'vectorstore') and legal_assistant.rag_system.vectorstore:
            try:
                collection = legal_assistant.rag_system.vectorstore._collection
                stats["documents_loaded"] = collection.count()
            except:
                stats["documents_loaded"] = "unknown"
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

def main():
    """Start the FastAPI server"""
    print("🚀 Starting AI Legal Assistant API Server...")
    print("📚 This may take a few minutes for initial setup...")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("🔍 Health check at: http://localhost:8000/health")
    
    uvicorn.run(
        "api_server:app",  # Import string for auto-reload
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable for production
        log_level="info"
    )

if __name__ == "__main__":
    main()
