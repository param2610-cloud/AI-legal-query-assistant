"""
Configuration settings for the AI Legal Assistant
===============================================
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "indian_code" / "acts"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
LOGS_DIR = BASE_DIR / "logs"

# Model configurations
DEFAULT_LLM_MODEL = "llama3.2:3b"
ALTERNATIVE_MODELS = ["mistral:7b", "codellama:7b", "llama2:7b"]

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEARCH_K = 5  # Number of documents to retrieve
MAX_ITERATIONS = 3  # For agent

# LLM parameters
TEMPERATURE = 0.1  # Low temperature for consistent legal advice
MAX_TOKENS = 2048

# Supported legal acts (for validation)
SUPPORTED_ACTS = [
    "consumer_act",
    "child_labour", 
    "civil_procedure",
    "adhaar_act",
    "birth_death_marraige_act",
    "dowry",
    "drug_cosmetics"
]

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "AI Legal Assistant",
    "page_icon": "⚖️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Legal disclaimer text
LEGAL_DISCLAIMER = """
⚠️ **IMPORTANT DISCLAIMER**

This AI Legal Assistant is for educational and informational purposes only. 
It does not constitute legal advice and should not be relied upon as a 
substitute for consultation with qualified legal professionals.

- Not a substitute for professional legal advice
- Consult a qualified lawyer for complex matters  
- Laws may change - verify current regulations
- Use at your own discretion

Always consult with a licensed attorney for complex legal matters, 
court proceedings, legal document preparation, or specific legal advice.
"""

# System prompts
LEGAL_SYSTEM_PROMPT = """You are an AI Legal Assistant specializing in Indian law. 
Your role is to help common people understand legal concepts in simple terms.

Guidelines:
1. Explain laws in plain, everyday language
2. Avoid complex legal jargon
3. Use examples when helpful
4. Provide practical guidance
5. Always remind users to consult lawyers for complex matters
6. Be accurate and cite relevant legal provisions
7. If unsure, say so and suggest professional consultation
"""

SITUATION_ANALYSIS_PROMPT = """Analyze the user's situation and identify relevant areas of Indian law.
Provide practical guidance while being clear about limitations of AI advice.

Focus on:
- Most applicable legal areas
- Key rights and protections
- Practical next steps
- When to seek professional help
"""
