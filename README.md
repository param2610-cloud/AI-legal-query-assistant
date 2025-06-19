# AI Legal Assistant - Agentic RAG System

## 🎯 Overview

This is a **free, open-source AI Legal Assistant** designed to help common people understand Indian laws in simple terms. It uses advanced RAG (Retrieval-Augmented Generation) technology with local LLMs to provide legal guidance without any API costs.

## ✨ Key Features

- **🆓 100% Free**: Uses Ollama for local LLM execution (no API costs)
- **🧠 Agentic RAG**: Intelligent document retrieval and contextual responses
- **📚 Comprehensive**: Covers multiple Indian legal acts
- **🌐 User-Friendly**: Web interface and CLI options
- **🔒 Privacy-First**: All processing happens locally
- **📱 Accessible**: Simple language explanations for common people

## 🏛️ Legal Acts Covered

- **Consumer Protection Act**
- **Child Labor Laws**
- **Civil Procedure Code**
- **Marriage Registration Laws**
- **Drug & Cosmetics Act**
- **Dowry Prohibition Act**
- **Birth, Death & Marriage Registration**
- **Aadhaar Act**

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- 8GB+ RAM (recommended for LLM)
- 10GB+ disk space

### 1. Automated Setup (Recommended)
```bash
# Make setup script executable and run
chmod +x setup.sh
./setup.sh
```

### 2. Manual Setup

#### Install Ollama (Local LLM)
```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Download a model (choose one)
ollama pull llama3.2:3b      # Recommended (2GB)
ollama pull mistral:7b       # Alternative (4GB)
ollama pull codellama:7b     # For code analysis (4GB)
```

#### Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application

#### Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```
Then open http://localhost:8501

#### Command Line Interface
```bash
python agent/legal_assistant.py
```

## 🎮 Usage Examples

### Example 1: Consumer Rights
**User**: "I bought a phone online but it's defective. The seller is refusing to refund. What are my rights?"

**AI Response**: 
- Explains Consumer Protection Act provisions
- Lists specific rights for online purchases
- Provides step-by-step complaint process
- Suggests time limits and documentation needed

### Example 2: Employment Issues
**User**: "My employer is making me work 12 hours daily and not paying overtime. Is this legal?"

**AI Response**:
- References labor law provisions
- Explains maximum working hours
- Details overtime payment requirements
- Suggests complaint mechanisms

### Example 3: Family Matters
**User**: "How do I register my marriage? What documents are needed?"

**AI Response**:
- Explains marriage registration process
- Lists required documents
- Provides step-by-step procedure
- Mentions state-specific variations

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Legal Agent    │───▶│  Response       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │   RAG System    │
                    └─────────────────┘
                               │
                    ┌─────────────────┐
                    │ Vector Database │
                    │  (ChromaDB)     │
                    └─────────────────┘
                               │
                    ┌─────────────────┐
                    │ Legal Documents │
                    │ (Processed JSON)│
                    └─────────────────┘
```

### Components:

1. **Legal Agent**: 
   - Analyzes user queries
   - Routes to appropriate tools
   - Maintains conversation context

2. **RAG System**:
   - Semantic search through legal documents
   - Context-aware retrieval
   - Document ranking and filtering

3. **Vector Database**:
   - ChromaDB for fast similarity search
   - Embeddings using Sentence Transformers
   - Persistent storage

4. **Local LLM**:
   - Ollama for model serving
   - Various model options (Llama, Mistral)
   - No API costs or external dependencies

## 🛠️ Technical Details

### Models Used
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: `llama3.2:3b` (default) or `mistral:7b`
- **Vector DB**: ChromaDB with cosine similarity

### Performance Optimizations
- Document chunking for better retrieval
- Caching of embeddings
- Optimized prompt templates
- Memory-efficient processing

### Privacy & Security
- All processing happens locally
- No data sent to external APIs
- User conversations not stored permanently
- Open-source and auditable

## 📊 Data Processing

The system processes legal documents through:

1. **JSON Parsing**: Extracts structured legal sections
2. **Text Chunking**: Splits large documents for better retrieval
3. **Embedding Generation**: Creates vector representations
4. **Indexing**: Stores in ChromaDB for fast search

## 🎯 Use Cases

### For Common People
- Understanding legal rights in simple terms
- Getting guidance on legal procedures
- Finding relevant laws for specific situations
- Learning about legal protections available

### For Legal Aid Organizations
- First-level screening of legal queries
- Educational resource for community outreach
- Standardized information delivery
- Multilingual legal guidance (future)

### For Students & Researchers
- Quick access to legal provisions
- Comparative analysis of different acts
- Legal research assistance
- Understanding legal frameworks

## 🚧 Limitations

- **Educational Purpose Only**: Not a substitute for professional legal advice
- **Data Currency**: Legal acts may change; system needs regular updates  
- **Complex Cases**: May not handle very specific or complex legal scenarios
- **Language**: Currently optimized for English; Hindi support planned
- **Jurisdictional**: Focused on Indian laws only

## 🛣️ Roadmap

### Phase 1 (Current)
- ✅ Basic RAG system
- ✅ Web interface
- ✅ Core legal acts
- ✅ Free LLM integration

### Phase 2 (Planned)
- 🔄 Hindi language support
- 🔄 More legal acts and regulations
- 🔄 Case law integration
- 🔄 Advanced agent capabilities

### Phase 3 (Future)
- 📋 Multi-modal support (images, documents)
- 📋 Voice interface
- 📋 Mobile app
- 📋 Integration with legal databases

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

1. **Legal Content**: Adding more acts and regulations
2. **Language Support**: Hindi and regional language translations
3. **UI/UX**: Improving the interface design
4. **Performance**: Optimization and scaling
5. **Testing**: Edge cases and accuracy improvements

## 📄 License

This project is open-source under the MIT License. See `LICENSE` file for details.

## ⚠️ Disclaimer

**Important**: This AI Legal Assistant is for educational and informational purposes only. It does not constitute legal advice and should not be relied upon as a substitute for consultation with qualified legal professionals. Laws and regulations may change, and specific situations may require specialized legal expertise.

Always consult with a licensed attorney for:
- Complex legal matters
- Court proceedings
- Legal document preparation
- Specific legal advice for your situation

## 📞 Support

- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the `docs/` folder for detailed guides

---

**🎉 Happy to help make Indian laws more accessible to everyone!**
