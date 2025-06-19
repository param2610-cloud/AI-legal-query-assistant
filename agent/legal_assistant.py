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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
# Import from correct locations for modern LangChain
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# Import LangGraph components for modern agent implementation
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
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
            
            # Look for various JSON file patterns
            json_files = []
            json_files.extend(list(act_dir.glob("*_sections*.json")))
            json_files.extend(list(act_dir.glob("*_structured.json")))
            json_files.extend(list(act_dir.glob("*_final*.json")))
            
            # Remove duplicates
            json_files = list(set(json_files))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Handle different JSON structures
                    if isinstance(data, dict):
                        # Case 1: Object with sections as keys (old format)
                        sections_found = False
                        for section_id, section_data in data.items():
                            if isinstance(section_data, dict) and ('title' in section_data or 'content' in section_data or 'text' in section_data):
                                sections_found = True
                                title = section_data.get('title', section_data.get('heading', f'Section {section_id}'))
                                content = section_data.get('content', section_data.get('text', ''))
                                
                                if content:  # Only add if there's actual content
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
                        
                        # Case 2: Object with 'sections' array (new format)
                        if not sections_found and 'sections' in data and isinstance(data['sections'], list):
                            sections = data['sections']
                            for i, section_data in enumerate(sections):
                                if isinstance(section_data, dict):
                                    section_id = section_data.get('section', section_data.get('section_number', str(i+1)))
                                    title = section_data.get('title', section_data.get('heading', f'Section {section_id}'))
                                    content = section_data.get('content', section_data.get('text', ''))
                                    
                                    if content:  # Only add if there's actual content
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
                    
                    # Case 3: Direct array of sections
                    elif isinstance(data, list):
                        for i, section_data in enumerate(data):
                            if isinstance(section_data, dict):
                                section_id = section_data.get('section', section_data.get('section_number', str(i+1)))
                                title = section_data.get('title', section_data.get('heading', f'Section {section_id}'))
                                content = section_data.get('content', section_data.get('text', ''))
                                
                                if content:  # Only add if there's actual content
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
                            elif isinstance(section_data, str) and section_data.strip():
                                # Handle simple string content
                                doc = Document(
                                    page_content=section_data,
                                    metadata={
                                        'act_name': act_name,
                                        'section_id': str(i+1),
                                        'title': f'Section {i+1}',
                                        'source_file': str(json_file),
                                        'document_type': 'legal_section'
                                    }
                                )
                                documents.append(doc)
                                
                except Exception as e:
                    logger.warning(f"Could not load {json_file}: {e}")
                    continue
        
        logger.info(f"Loaded {len(documents)} legal documents")
        return documents

class LegalRAGSystem:
    """RAG system specifically designed for legal queries"""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        # Use modern HuggingFace embeddings
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            except ImportError:
                from langchain.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
        
        # Use ChatOllama instead of OllamaLLM for tool calling support
        self.llm = ChatOllama(model=model_name, temperature=0.1)
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
        
        # Create a custom QA chain using modern patterns
        from langchain_core.runnables import RunnableParallel, RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        legal_prompt = PromptTemplate.from_template("""
You are an AI Legal Assistant specializing in Indian laws. Your role is to help common people understand legal matters in simple, practical terms.

Context from Legal Documents:
{context}

User Question: {question}

Response Guidelines:
- Start with a direct answer to their question
- Explain relevant legal provisions in plain English
- Use bullet points for clarity
- Include practical next steps if applicable
- Mention if they should consult a lawyer for complex situations

Response:""")
        
        # Modern chain using LCEL (LangChain Expression Language)
        self.qa_chain = (
            RunnableParallel({
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            })
            | legal_prompt
            | self.llm
            | StrOutputParser()
        )

class LegalAgent:
    """Modern Agentic system for legal assistance using LangGraph"""
    
    def __init__(self, rag_system: LegalRAGSystem):
        self.rag_system = rag_system
        self.graph = None
        self.setup_graph()
    
    def legal_search_tool(self, query: str) -> str:
        """Tool for searching legal documents"""
        try:
            result = self.rag_system.qa_chain.invoke(query)
            return result
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
            response = self.rag_system.llm.invoke(analysis_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error analyzing situation: {e}"
    
    def generate_query_or_respond(self, state: MessagesState):
        """Decide whether to use tools or respond directly"""
        
        # Create legal search tool using modern @tool decorator
        @tool
        def legal_document_search(query: str) -> str:
            """Search through Indian legal acts and regulations to find relevant information.
            
            Args:
                query: The search query about legal matters
                
            Returns:
                Relevant legal information and guidance
            """
            try:
                return self.rag_system.qa_chain.invoke(query)
            except Exception as e:
                return f"Error searching legal documents: {e}"
        
        # Get the latest message
        messages = state["messages"]
        
        # Use LLM to decide if tools are needed
        response = self.rag_system.llm.bind_tools([legal_document_search]).invoke(messages)
        
        return {"messages": [response]}
    
    def grade_documents(self, state: MessagesState):
        """Grade document relevance"""
        question = state["messages"][0].content
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'content'):
            context = last_message.content
        else:
            context = str(last_message)
        
        grade_prompt = f"""
        You are grading document relevance. 
        Question: {question}
        Document: {context}
        
        Is this document relevant to answering the question? 
        Answer with ONLY 'yes' or 'no' and nothing else.
        """
        
        try:
            # Try structured output first
            try:
                from pydantic import BaseModel, Field
                
                class GradeDocuments(BaseModel):
                    binary_score: str = Field(description="Relevance score: 'yes' or 'no'")
                
                response = self.rag_system.llm.with_structured_output(GradeDocuments).invoke(
                    [{"role": "user", "content": grade_prompt}]
                )
                score = response.binary_score.lower()
            except:
                # Fallback to simple text response
                response = self.rag_system.llm.invoke([{"role": "user", "content": grade_prompt}])
                score = response.content.lower().strip() if hasattr(response, 'content') else str(response).lower().strip()
            
            if "yes" in score:
                return "generate_answer"
            else:
                return "rewrite_question"
        except Exception as e:
            logger.warning(f"Document grading failed: {e}, defaulting to generate_answer")
            # Default to generating answer if grading fails
            return "generate_answer"
    
    def rewrite_question(self, state: MessagesState):
        """Rewrite unclear questions"""
        messages = state["messages"]
        original_question = messages[0].content
        
        rewrite_prompt = f"""
        The original question may need clarification for better search results.
        Original question: {original_question}
        
        Rewrite this question to be more specific and searchable for Indian legal documents:
        """
        
        response = self.rag_system.llm.invoke([{"role": "user", "content": rewrite_prompt}])
        
        # Replace the original question with the rewritten one
        return {"messages": [HumanMessage(content=response.content)]}
    
    def generate_answer(self, state: MessagesState):
        """Generate final answer using retrieved context"""
        question = state["messages"][0].content
        
        # Find the tool message with context
        context = ""
        for msg in state["messages"]:
            if hasattr(msg, 'name') and msg.name == "legal_document_search":
                context = msg.content
                break
            elif hasattr(msg, 'content') and len(msg.content) > 100:
                context = msg.content
        
        answer_prompt = f"""
        You are an AI Legal Assistant for Indian law. Answer the question using the provided context.
        
        Question: {question}
        Context: {context}
        
        Provide a helpful answer that:
        - Explains the law in simple terms
        - Gives practical guidance
        - Uses bullet points for clarity
        - Mentions when to consult a lawyer
        
        Keep it concise and user-friendly.
        """
        
        response = self.rag_system.llm.invoke([{"role": "user", "content": answer_prompt}])
        return {"messages": [AIMessage(content=response.content)]}
    
    def setup_graph(self):
        """Create the LangGraph workflow"""
        
        # Create legal search tool using modern @tool decorator
        @tool
        def legal_document_search(query: str) -> str:
            """Search through Indian legal acts and regulations.
            
            Args:
                query: The search query about legal matters
                
            Returns:
                Relevant legal information and guidance
            """
            try:
                return self.rag_system.qa_chain.invoke(query)
            except Exception as e:
                return f"Error searching legal documents: {e}"
        
        # Create the graph
        workflow = StateGraph(MessagesState)
        
        # Add nodes
        workflow.add_node("generate_query_or_respond", self.generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([legal_document_search]))
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
        
        # Compile the graph
        self.graph = workflow.compile()
    
    def invoke(self, input_dict):
        """Process user input through the graph"""
        try:
            if isinstance(input_dict.get("input"), str):
                messages = [HumanMessage(content=input_dict["input"])]
            else:
                messages = input_dict.get("messages", [])
            
            # Run the graph
            result = None
            for chunk in self.graph.stream({"messages": messages}):
                for node, update in chunk.items():
                    result = update
            
            if result and "messages" in result:
                final_message = result["messages"][-1]
                if hasattr(final_message, 'content'):
                    return {"output": final_message.content}
                else:
                    return {"output": str(final_message)}
            else:
                return {"output": "No response generated"}
                
        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            # Fallback to direct RAG
            try:
                query = input_dict.get("input", "")
                if not query and "messages" in input_dict:
                    query = input_dict["messages"][-1].content if input_dict["messages"] else ""
                
                rag_result = self.rag_system.qa_chain.invoke({"query": query})
                return {"output": rag_result.get("result", "No response generated")}
            except Exception as fallback_error:
                return {"output": f"Error: {str(e)} | Fallback error: {str(fallback_error)}"}

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
            response = self.agent.invoke({"input": full_query})
            
            return {
                "question": question,
                "response": response.get("output", "No response generated"),
                "context": context,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            # Fallback to direct RAG query if agent fails
            try:
                logger.info("Falling back to direct RAG query...")
                rag_result = self.rag_system.qa_chain.invoke({"query": question})
                return {
                    "question": question,
                    "response": rag_result.get("result", "No response generated"),
                    "context": context,
                    "status": "success_fallback",
                    "source_documents": rag_result.get("source_documents", [])
                }
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return {
                    "question": question,
                    "error": f"Primary error: {str(e)}, Fallback error: {str(fallback_error)}",
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
