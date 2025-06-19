# Modern AI Legal Assistant - Agentic RAG System

A comprehensive legal assistant built with modern LangChain/LangGraph patterns to help common people understand Indian laws using **completely free and open-source tools** (no paid API keys required).

## 🎯 Key Features

### ✅ **No Paid APIs Required**
- Uses **Ollama** for local LLM (llama3.2:3b or mistral)
- **Sentence Transformers** for embeddings 
- **ChromaDB** for vector storage
- Completely free and runs offline after setup

### 🤖 **Modern Agentic RAG Architecture**
- **LangGraph-based** workflow for intelligent decision making
- **Tool calling** with ChatOllama for dynamic responses
- **Document grading** and query rewriting
- **Fallback mechanisms** for reliability

### 📚 **Legal Domain Expertise**
- Specialized for **Indian legal acts**
- Processes multiple legal document formats
- Provides **plain language explanations**
- Context-aware legal guidance

### 🔧 **Updated Technology Stack**
- Modern **LangChain v0.2+** patterns
- No deprecated imports or warnings
- **Type-safe** with Pydantic models
- Comprehensive error handling

## 🚀 Quick Start

### 1. Setup Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install and setup Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llama3.2:3b
```

### 2. Run the Modern Assistant

```bash
# Test the system
python test_modern_assistant.py --mode test

# Interactive mode
python test_modern_assistant.py --mode interactive
```

### 3. Use in Your Code

```python
from agent.modern_legal_assistant import ModernLegalAssistant

# Initialize
assistant = ModernLegalAssistant(
    model_name="llama3.2:3b",
    data_dir="indian_code/acts"
)

# Setup the system
if assistant.initialize():
    # Ask questions
    result = assistant.ask_question(
        "What are my consumer rights in India?"
    )
    print(result["response"])
```

## 🏗️ Architecture Overview

### Agentic RAG Workflow

The system uses a sophisticated workflow that:

1. **Analyzes** user questions for legal context
2. **Decides** whether to search legal documents or respond directly
3. **Searches** relevant legal acts when needed
4. **Grades** document relevance to the question
5. **Rewrites** questions if documents aren't relevant
6. **Generates** comprehensive answers in simple language

### Key Components

1. **LegalDataProcessor**: Handles multiple JSON formats from Indian legal acts
2. **ModernLegalAssistant**: Main orchestrator with LangGraph workflow
3. **Tool System**: Dynamic legal document search with context awareness
4. **Grading System**: Evaluates document relevance and query quality
5. **Fallback Mechanisms**: Ensures reliability even when components fail

## 📋 Supported Legal Documents

The system currently processes these Indian legal acts:

- **Consumer Protection Act**
- **Child Labour (Prohibition and Regulation) Act**
- **Civil Procedure Code**
- **Aadhaar Act**
- **Birth, Death and Marriage Registration Act**
- **Dowry Prohibition Act**
- **Drug and Cosmetics Act**

### Document Format Support

- `*_sections_only.json` - Section-based structure
- `*_structured.json` - Hierarchical organization  
- `*_final.json` - Processed legal content
- Nested JSON objects and arrays
- Multiple metadata fields

## 🔍 Example Interactions

### Consumer Rights Query
```
User: "What should I do if a product I bought is defective?"
