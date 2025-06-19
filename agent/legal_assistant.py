"""
AI Legal Assistant - Agentic RAG System
=======================================

This system helps common people understand Indian legal acts through:
1. Vector-based retrieval of relevant legal sections
2. Context-aware legal interpretation
3. Plain language explanations
4. Situation-specific legal guidance

Key Features:
- Free/Open Source (uses Ollama for LLM)
- Multi-language support for Indian laws
- Context-aware responses
- Simplified legal explanations
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDataProcessor:
    """Process and prepare legal documents for RAG system"""
    
    def __init__(self, data_dir: str = "indian_code/acts"):
        self.data_dir = Path(data_dir)
        self.documents = []
        self.metadata = []
    
    def load_legal_acts(self) -> List[Document]:
        """Load all legal acts from JSON files"""
        documents = []
        
        # Get all act directories
        act_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        for act_dir in tqdm(act_dirs, desc="Loading legal acts"):
            act_name = act_dir.name.replace('_', ' ').title()
            
            # Look for structured JSON files
            json_files = list(act_dir.glob("*_sections*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        sections = json.load(f)
                    
                    for section_id, section_data in sections.items():
                        if isinstance(section_data, dict):
                            title = section_data.get('title', f'Section {section_id}')
                            content = section_data.get('content', '')
                            
                            # Create document with metadata
                            doc = Document(
                                page_content=f"Title: {title}\n\nContent: {content}",
                                metadata={
                                    'act_name': act_name,
                                    'section_id': section_id,
                                    'title': title,
                                    'source_file': str(json_file),
                                    'document_type': 'legal_section'
                                }
                            )
                            documents.append(doc)
                            
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {len(documents)} legal documents")
        return documents

class LegalRAGSystem:
    """RAG system specifically designed for legal queries"""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm = OllamaLLM(model=model_name, temperature=0.1)
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        
    def setup_vectorstore(self, documents: List[Document]):
        """Set up ChromaDB vector store"""
        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(split_docs)} chunks")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
    def create_legal_qa_chain(self):
        """Create specialized QA chain for legal queries"""
        
        legal_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an AI Legal Assistant specializing in Indian law. Your role is to help common people understand legal concepts in simple terms.

Given the following legal context and a user's question, provide a helpful response that:
1. Explains the relevant law in simple, everyday language
2. Highlights key points that apply to the user's situation
3. Provides practical guidance
4. Uses examples when helpful
5. Avoids complex legal jargon

Legal Context:
{context}

User Question: {question}

Response Guidelines:
- Start with a direct answer to their question
- Explain relevant legal provisions in plain English
- Use bullet points for clarity
- Include practical next steps if applicable
- Mention if they should consult a lawyer for complex situations

Response:"""
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": legal_prompt},
            return_source_documents=True
        )

class LegalAgent:
    """Agentic system for legal assistance"""
    
    def __init__(self, rag_system: LegalRAGSystem):
        self.rag_system = rag_system
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.agent = None
        self.setup_agent()
    
    def legal_search_tool(self, query: str) -> str:
        """Tool for searching legal documents"""
        try:
            result = self.rag_system.qa_chain({"query": query})
            return result["result"]
        except Exception as e:
            return f"Error searching legal documents: {e}"
    
    def situation_analyzer_tool(self, situation: str) -> str:
        """Analyze user's situation and suggest relevant legal areas"""
        
        analysis_prompt = f"""
        Analyze this person's situation and identify which areas of Indian law might be relevant:
        
        Situation: {situation}
        
        Consider these legal areas:
        - Consumer Protection
        - Child Labor Laws  
        - Civil Procedure
        - Marriage and Family Laws
        - Property Laws
        - Employment Laws
        - Criminal Laws
        
        Provide:
        1. Most relevant legal areas (2-3)
        2. Specific laws that might apply
        3. Key questions they should consider
        4. Recommended next steps
        
        Keep the response practical and easy to understand.
        """
        
        try:
            response = self.rag_system.llm(analysis_prompt)
            return response
        except Exception as e:
            return f"Error analyzing situation: {e}"
    
    def setup_agent(self):
        """Set up the legal agent with tools"""
        
        tools = [
            Tool(
                name="Legal Document Search",
                func=self.legal_search_tool,
                description="Search through Indian legal acts and regulations to find relevant information. Use this when the user asks about specific laws, rights, or legal procedures."
            ),
            Tool(
                name="Situation Analyzer",
                func=self.situation_analyzer_tool,
                description="Analyze a person's specific situation to identify relevant legal areas and provide guidance. Use this when the user describes their personal situation or problem."
            )
        ]
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.rag_system.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            max_iterations=3
        )

class SimpleLegalAssistant:
    """Main interface for the legal assistant"""
    
    def __init__(self):
        self.processor = LegalDataProcessor()
        self.rag_system = LegalRAGSystem()
        self.agent = None
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the system"""
        try:
            logger.info("Initializing Legal Assistant...")
            
            # Load legal documents
            documents = self.processor.load_legal_acts()
            if not documents:
                logger.error("No legal documents found!")
                return False
            
            # Setup RAG system
            self.rag_system.setup_vectorstore(documents)
            self.rag_system.create_legal_qa_chain()
            
            # Setup agent
            self.agent = LegalAgent(self.rag_system)
            
            self.is_initialized = True
            logger.info("Legal Assistant initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    def ask_question(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Ask a legal question"""
        if not self.is_initialized:
            return {"error": "System not initialized. Please call initialize() first."}
        
        try:
            # Prepare the query
            if context:
                full_query = f"Context: {context}\n\nQuestion: {question}"
            else:
                full_query = question
            
            # Get response from agent
            response = self.agent.agent.run(full_query)
            
            return {
                "question": question,
                "response": response,
                "context": context,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "error": str(e),
                "status": "error"
            }
    
    def get_relevant_acts(self, topic: str) -> List[Dict]:
        """Get acts relevant to a specific topic"""
        if not self.is_initialized:
            return []
        
        try:
            # Search for relevant documents
            docs = self.rag_system.vectorstore.similarity_search(topic, k=10)
            
            # Group by act
            acts = {}
            for doc in docs:
                act_name = doc.metadata.get('act_name', 'Unknown Act')
                if act_name not in acts:
                    acts[act_name] = {
                        'act_name': act_name,
                        'relevant_sections': [],
                        'relevance_score': 0
                    }
                
                acts[act_name]['relevant_sections'].append({
                    'section_id': doc.metadata.get('section_id'),
                    'title': doc.metadata.get('title'),
                    'content_preview': doc.page_content[:200] + "..."
                })
            
            return list(acts.values())
            
        except Exception as e:
            logger.error(f"Error getting relevant acts: {e}")
            return []

# Example usage and testing
if __name__ == "__main__":
    # Initialize the system
    assistant = SimpleLegalAssistant()
    
    if assistant.initialize():
        print("🎉 Legal Assistant is ready!")
        print("\nExample questions you can ask:")
        print("1. What are my rights as a consumer?")
        print("2. My employer is not paying minimum wage. What can I do?")
        print("3. I want to understand child labor laws in India")
        print("4. What is the process for civil court procedures?")
        
        # Interactive mode
        while True:
            print("\n" + "="*50)
            user_input = input("\nAsk your legal question (or 'quit' to exit): ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            print("\n🔍 Processing your question...")
            result = assistant.ask_question(user_input)
            
            if result["status"] == "success":
                print(f"\n📋 Answer:\n{result['response']}")
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
    
    else:
        print("❌ Failed to initialize Legal Assistant")
