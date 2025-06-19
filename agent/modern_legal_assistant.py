"""
Modern AI Legal Assistant - Agentic RAG System
=============================================

A comprehensive legal assistant using modern LangChain/LangGraph patterns
for helping common people understand Indian laws.

Features:
- Modern LangGraph-based agentic architecture
- ChatOllama for tool calling support
- Proper error handling and fallbacks
- Updated imports to avoid deprecation warnings
- Free/open source (no paid API keys required)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path

# Modern LangChain imports
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# LangGraph for modern agent implementation
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Embeddings with proper fallback
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings

from pydantic import BaseModel, Field
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModernLegalAssistant:
    """Modern Legal Assistant with Agentic RAG capabilities"""
    
    def __init__(self, model_name: str = "llama3.2:3b", data_dir: str = "indian_code/acts"):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        
        # Initialize components
        self.embeddings = self._setup_embeddings()
        self.llm = ChatOllama(model=model_name, temperature=0.1)
        self.vectorstore = None
        self.retriever = None
        self.graph = None
        
        logger.info(f"Initialized with model: {model_name}")
    
    def _setup_embeddings(self):
        """Setup embeddings with proper fallback"""
        try:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            logger.warning(f"Embeddings setup warning: {e}")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def load_legal_documents(self) -> List[Document]:
        """Load and process legal documents"""
        documents = []
        
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            return documents
        
        act_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        for act_dir in tqdm(act_dirs, desc="Loading legal acts"):
            act_name = act_dir.name.replace('_', ' ').title()
            
            # Find JSON files
            json_files = []
            json_files.extend(list(act_dir.glob("*_sections*.json")))
            json_files.extend(list(act_dir.glob("*_structured.json")))
            json_files.extend(list(act_dir.glob("*_final*.json")))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    docs = self._process_legal_json(data, act_name, json_file)
                    documents.extend(docs)
                    
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")
                    continue
        
        logger.info(f"Loaded {len(documents)} legal documents")
        return documents
    
    def _process_legal_json(self, data: Any, act_name: str, source_file: Path) -> List[Document]:
        """Process different JSON structures"""
        documents = []
        
        if isinstance(data, dict):
            # Handle structured format
            if 'sections' in data and isinstance(data['sections'], list):
                for i, section in enumerate(data['sections']):
                    doc = self._create_document_from_section(
                        section, act_name, source_file, i
                    )
                    if doc:
                        documents.append(doc)
            else:
                # Handle object with sections as keys
                for section_id, section_data in data.items():
                    if isinstance(section_data, dict):
                        doc = self._create_document_from_section(
                            section_data, act_name, source_file, section_id
                        )
                        if doc:
                            documents.append(doc)
        
        elif isinstance(data, list):
            # Handle array format
            for i, section in enumerate(data):
                if isinstance(section, dict):
                    doc = self._create_document_from_section(
                        section, act_name, source_file, i
                    )
                    if doc:
                        documents.append(doc)
                elif isinstance(section, str) and section.strip():
                    documents.append(Document(
                        page_content=section.strip(),
                        metadata={
                            'act_name': act_name,
                            'section_id': str(i+1),
                            'title': f'Section {i+1}',
                            'source_file': str(source_file),
                            'document_type': 'legal_section'
                        }
                    ))
        
        return documents
    
    def _create_document_from_section(self, section_data: Dict, act_name: str, 
                                    source_file: Path, section_id: Any) -> Optional[Document]:
        """Create a document from section data"""
        try:
            # Extract content
            title = section_data.get('title', section_data.get('heading', f'Section {section_id}'))
            content = section_data.get('content', section_data.get('text', ''))
            
            if not content or not content.strip():
                return None
            
            # Format content
            formatted_content = f"Title: {title}\\n\\nContent: {content}"
            
            return Document(
                page_content=formatted_content,
                metadata={
                    'act_name': act_name,
                    'section_id': str(section_id),
                    'title': title,
                    'source_file': str(source_file),
                    'document_type': 'legal_section'
                }
            )
        except Exception as e:
            logger.warning(f"Error creating document from section: {e}")
            return None
    
    def setup_vectorstore(self, documents: List[Document]):
        """Setup ChromaDB vectorstore"""
        if not documents:
            logger.error("No documents to process")
            return False
        
        try:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\\n\\n", "\\n", ".", "!", "?", ",", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(split_docs)} chunks")
            
            # Create vectorstore
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
            
            logger.info("Vectorstore setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Vectorstore setup failed: {e}")
            return False
    
    def create_legal_tools(self):
        """Create tools for the agent"""
        
        @tool
        def legal_document_search(query: str) -> str:
            """Search through Indian legal acts and regulations to find relevant information.
            
            Args:
                query: The search query about legal matters
                
            Returns:
                Relevant legal information and guidance in simple language
            """
            try:
                if not self.retriever:
                    return "Legal database not available. Please initialize the system first."
                
                # Get relevant documents
                docs = self.retriever.invoke(query)
                
                if not docs:
                    return "No relevant legal information found for your query."
                
                # Format context
                context = "\\n\\n".join([doc.page_content for doc in docs])
                
                # Create response using LLM
                prompt = PromptTemplate.from_template("""
You are an AI Legal Assistant specializing in Indian law. Help common people understand legal matters.

Legal Context:
{context}

User Query: {query}

Provide a helpful response that:
- Explains the relevant law in simple, everyday language
- Highlights key points that apply to the user's situation  
- Provides practical guidance
- Uses bullet points for clarity
- Mentions when to consult a lawyer for complex situations

Response:""")
                
                formatted_prompt = prompt.format(context=context, query=query)
                response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
                
                return response.content if hasattr(response, 'content') else str(response)
                
            except Exception as e:
                logger.error(f"Legal search error: {e}")
                return f"Error searching legal documents: {e}"
        
        return [legal_document_search]
    
    def generate_query_or_respond(self, state: MessagesState):
        """Generate query or respond directly"""
        tools = self.create_legal_tools()
        
        # Get the latest message
        messages = state["messages"]
        
        # Use LLM with tools to decide
        response = self.llm.bind_tools(tools).invoke(messages)
        
        return {"messages": [response]}
    
    def grade_documents(self, state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        """Grade document relevance"""
        try:
            question = state["messages"][0].content
            last_message = state["messages"][-1]
            
            # Extract context from tool message
            context = ""
            if hasattr(last_message, 'content'):
                context = last_message.content
            else:
                context = str(last_message)
            
            # Create grading model
            class GradeDocuments(BaseModel):
                binary_score: str = Field(description="Relevance score: 'yes' or 'no'")
            
            grade_prompt = f"""
Grade the relevance of the retrieved document to the user question.

Question: {question}
Document: {context[:500]}...

Is this document relevant to answering the question?
Answer with 'yes' or 'no' only.
"""
            
            try:
                response = self.llm.with_structured_output(GradeDocuments).invoke(
                    [HumanMessage(content=grade_prompt)]
                )
                score = response.binary_score.lower()
            except:
                # Fallback to simple response
                response = self.llm.invoke([HumanMessage(content=grade_prompt)])
                content = response.content if hasattr(response, 'content') else str(response)
                score = content.lower().strip()
            
            if "yes" in score:
                return "generate_answer"
            else:
                return "rewrite_question"
                
        except Exception as e:
            logger.warning(f"Document grading failed: {e}")
            return "generate_answer"  # Default to generating answer
    
    def rewrite_question(self, state: MessagesState):
        """Rewrite unclear questions"""
        messages = state["messages"]
        original_question = messages[0].content
        
        rewrite_prompt = f"""
Rewrite this question to be more specific and searchable for Indian legal documents:

Original question: {original_question}

Rewritten question (be specific about Indian law context):"""
        
        response = self.llm.invoke([HumanMessage(content=rewrite_prompt)])
        content = response.content if hasattr(response, 'content') else str(response)
        
        return {"messages": [HumanMessage(content=content)]}
    
    def generate_answer(self, state: MessagesState):
        """Generate final answer"""
        question = state["messages"][0].content
        
        # Find context from tool messages
        context = ""
        for msg in state["messages"]:
            if hasattr(msg, 'name') and msg.name == "legal_document_search":
                context = msg.content
                break
            elif hasattr(msg, 'content') and len(str(msg.content)) > 100:
                context = str(msg.content)
        
        if not context:
            context = "No specific legal context found."
        
        answer_prompt = f"""
You are an AI Legal Assistant for Indian law. Provide a comprehensive answer.

Question: {question}
Legal Context: {context}

Provide a helpful answer that:
- Starts with a direct answer
- Explains relevant laws in simple terms
- Gives practical guidance
- Uses bullet points for clarity
- Mentions when to consult a lawyer

Keep it concise and user-friendly.
"""
        
        response = self.llm.invoke([HumanMessage(content=answer_prompt)])
        content = response.content if hasattr(response, 'content') else str(response)
        
        return {"messages": [AIMessage(content=content)]}
    
    def setup_graph(self):
        """Setup the LangGraph workflow"""
        try:
            tools = self.create_legal_tools()
            
            # Create workflow
            workflow = StateGraph(MessagesState)
            
            # Add nodes
            workflow.add_node("generate_query_or_respond", self.generate_query_or_respond)
            workflow.add_node("retrieve", ToolNode(tools))
            workflow.add_node("rewrite_question", self.rewrite_question)
            workflow.add_node("generate_answer", self.generate_answer)
            
            # Add edges
            workflow.add_edge(START, "generate_query_or_respond")
            
            # Conditional edges
            workflow.add_conditional_edges(
                "generate_query_or_respond",
                tools_condition,
                {
                    "tools": "retrieve",
                    END: END,
                },
            )
            
            workflow.add_conditional_edges(
                "retrieve",
                self.grade_documents,
            )
            
            workflow.add_edge("generate_answer", END)
            workflow.add_edge("rewrite_question", "generate_query_or_respond")
            
            # Compile
            self.graph = workflow.compile()
            logger.info("Graph setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Graph setup failed: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize the complete system"""
        try:
            logger.info("Initializing Modern Legal Assistant...")
            
            # Load documents
            documents = self.load_legal_documents()
            if not documents:
                logger.error("No legal documents found!")
                return False
            
            # Setup vectorstore
            if not self.setup_vectorstore(documents):
                return False
            
            # Setup graph
            if not self.setup_graph():
                return False
            
            logger.info("🎉 Modern Legal Assistant initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a legal question"""
        if not self.graph:
            return {
                "error": "System not initialized. Please call initialize() first.",
                "status": "error"
            }
        
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Create input
            input_data = {"messages": [HumanMessage(content=question)]}
            
            # Run through graph
            result = None
            for chunk in self.graph.stream(input_data):
                for node, update in chunk.items():
                    result = update
                    logger.debug(f"Node {node}: {type(update)}")
            
            # Extract final response
            if result and "messages" in result:
                final_message = result["messages"][-1]
                response_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
                
                return {
                    "question": question,
                    "response": response_content,
                    "status": "success"
                }
            else:
                return {
                    "question": question,
                    "error": "No response generated from graph",
                    "status": "error"
                }
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            
            # Fallback to direct search
            try:
                logger.info("Attempting fallback search...")
                tools = self.create_legal_tools()
                if tools:
                    response = tools[0].invoke({"query": question})
                    return {
                        "question": question,
                        "response": response,
                        "status": "success_fallback"
                    }
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {fallback_error}")
            
            return {
                "question": question,
                "error": str(e),
                "status": "error"
            }
    
    def get_relevant_acts(self, topic: str) -> List[Dict]:
        """Get acts relevant to a topic"""
        if not self.vectorstore:
            return []
        
        try:
            docs = self.vectorstore.similarity_search(topic, k=10)
            
            acts = {}
            for doc in docs:
                act_name = doc.metadata.get('act_name', 'Unknown Act')
                if act_name not in acts:
                    acts[act_name] = {
                        'act_name': act_name,
                        'relevant_sections': []
                    }
                
                acts[act_name]['relevant_sections'].append({
                    'section_id': doc.metadata.get('section_id'),
                    'title': doc.metadata.get('title'),
                    'preview': doc.page_content[:200] + "..."
                })
            
            return list(acts.values())
            
        except Exception as e:
            logger.error(f"Error getting relevant acts: {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Initialize
    assistant = ModernLegalAssistant()
    
    if assistant.initialize():
        print("🎉 Modern Legal Assistant is ready!")
        print("\\nExample questions:")
        print("1. What are my consumer rights?")
        print("2. What should I do if my employer doesn't pay minimum wage?")
        print("3. Tell me about child labor laws in India")
        
        # Interactive mode
        while True:
            print("\\n" + "="*50)
            question = input("\\nAsk your legal question (or 'quit' to exit): ")
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            print("\\n🔍 Processing...")
            result = assistant.ask_question(question)
            
            if result["status"] == "success":
                print(f"\\n📋 Answer:\\n{result['response']}")
            elif result["status"] == "success_fallback":
                print(f"\\n📋 Answer (fallback):\\n{result['response']}")
            else:
                print(f"\\n❌ Error: {result.get('error', 'Unknown error')}")
    else:
        print("❌ Failed to initialize Legal Assistant")
