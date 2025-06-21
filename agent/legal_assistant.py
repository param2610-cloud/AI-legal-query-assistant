"""
AI Legal Assistant - Agentic RAG System
=======================================

This system helps common people understand Indian legal acts through:
1. Vector-based retrieval of relevant legal sections
2. Context-aware legal interpretation
3. Plain language explanations
4. Situation-specific legal guidance
5. Live case law search via Indian Kanoon API

Key Features:
- Free/Open Source (uses Ollama for LLM)
- Multi-language support for Indian laws
- Context-aware responses
- Simplified legal explanations
- Live case law database integration
"""

import os
import sys
import json
import logging
import re
import time
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from tqdm import tqdm
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

# Indian Kanoon API Integration - Using Clean Production Client
try:
    import sys
    from pathlib import Path
    # Add the project root to path to find indian_kanoon_client
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from indian_kanoon_client import IndianKanoonClient, create_indian_kanoon_client
    INDIAN_KANOON_AVAILABLE = True
    logger.info("Indian Kanoon API integration loaded successfully")
except ImportError as e:
    logger.warning(f"Indian Kanoon API not available: {e}")
    INDIAN_KANOON_AVAILABLE = False

# Import enhanced case law analyzer
try:
    # Try relative import first
    from .enhanced_case_law_analyzer import CaseLawAnalyzer, analyze_legal_case, find_similar_precedents
    CASE_LAW_ANALYZER_AVAILABLE = True
    logger.info("Enhanced case law analyzer loaded successfully")
except ImportError:
    try:
        # Try absolute import from agent package
        from agent.enhanced_case_law_analyzer import CaseLawAnalyzer, analyze_legal_case, find_similar_precedents
        CASE_LAW_ANALYZER_AVAILABLE = True
        logger.info("Enhanced case law analyzer loaded successfully")
    except ImportError:
        try:
            # Try direct import (for streamlit)
            import sys
            from pathlib import Path
            current_dir = Path(__file__).parent
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            from enhanced_case_law_analyzer import CaseLawAnalyzer, analyze_legal_case, find_similar_precedents
            CASE_LAW_ANALYZER_AVAILABLE = True
            logger.info("Enhanced case law analyzer loaded successfully")
        except ImportError as e:
            logger.warning(f"Enhanced case law analyzer not available: {e}")
            CASE_LAW_ANALYZER_AVAILABLE = False


class LegalTermsIntegrator:
    """
    Integrates legal terms dictionary with RAG system for better context
    """
    
    def __init__(self, terms_dir: str = "legal_terms"):
        self.terms_dir = Path(terms_dir)
        self.terms_data = {}
        self.term_definitions = {}
        self.term_variants = {}
        self.keywords_index = {}
        self.terms_by_category = {}
        
        self._load_legal_terms()
    
    def _load_legal_terms(self) -> bool:
        """Load legal terms from JSON files"""
        try:
            # Load structured data for comprehensive features
            structured_file = self.terms_dir / "legal_terms_structured.json"
            if structured_file.exists():
                with open(structured_file, 'r', encoding='utf-8') as f:
                    self.terms_data = json.load(f)
                
                # Extract useful indexes
                self.term_definitions = self.terms_data.get("term_definitions", {})
                self.term_variants = self.terms_data.get("term_variants", {})
                self.keywords_index = self.terms_data.get("keywords_index", {})
                self.terms_by_category = self.terms_data.get("terms_by_category", {})
                
                logger.info(f"Loaded {len(self.term_definitions)} legal terms")
                return True
            
            # Fallback to dictionary file
            dict_file = self.terms_dir / "legal_terms_dictionary.json"
            if dict_file.exists():
                with open(dict_file, 'r', encoding='utf-8') as f:
                    self.term_definitions = json.load(f)
                
                logger.info(f"Loaded {len(self.term_definitions)} legal terms (fallback)")
                return True
                
            logger.warning("No legal terms files found")
            return False
            
        except Exception as e:
            logger.error(f"Error loading legal terms: {e}")
            return False
    
    def extract_legal_terms_from_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Extract legal terms from user query and provide their definitions
        
        Args:
            query: User query text
            
        Returns:
            List of dictionaries containing term info and definitions
        """
        found_terms = []
        query_words = self._tokenize_query(query)
        
        # Direct term matching
        found_terms.extend(self._match_direct_terms(query, query_words))
        
        # Variant matching for better coverage
        found_terms.extend(self._match_variant_terms(query, query_words))
        
        # Keyword-based matching for related terms
        found_terms.extend(self._match_keyword_terms(query_words))
        
        # Remove duplicates while preserving order
        unique_terms = self._deduplicate_terms(found_terms)
        
        logger.info(f"Found {len(unique_terms)} legal terms in query")
        return unique_terms
    
    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize query into words and phrases"""
        # Clean and normalize query
        query_clean = re.sub(r'[^\w\s\-]', ' ', query.lower())
        words = query_clean.split()
        
        # Generate multi-word phrases (2-4 words)
        phrases = []
        for i in range(len(words)):
            for j in range(2, min(5, len(words) - i + 1)):
                phrases.append(' '.join(words[i:i+j]))
        
        return words + phrases
    
    def _match_direct_terms(self, query: str, query_words: List[str]) -> List[Dict[str, Any]]:
        """Match direct terms from dictionary"""
        matches = []
        
        # Check exact matches in definitions
        for term, definition in self.term_definitions.items():
            if term.lower() in query.lower():
                matches.append({
                    'term': term,
                    'definition': definition,
                    'confidence': 1.0,
                    'match_type': 'exact',
                    'category': self._get_term_category(term)
                })
        
        return matches
    
    def _match_variant_terms(self, query: str, query_words: List[str]) -> List[Dict[str, Any]]:
        """Match term variants"""
        matches = []
        
        if not self.term_variants:
            return matches
        
        query_lower = query.lower()
        
        for variant, original_term in self.term_variants.items():
            if variant.lower() in query_lower:
                definition = self.term_definitions.get(original_term.lower(), "No definition available")
                matches.append({
                    'term': original_term,
                    'definition': definition,
                    'confidence': 0.8,
                    'match_type': 'variant',
                    'matched_variant': variant,
                    'category': self._get_term_category(original_term)
                })
        
        return matches
    
    def _match_keyword_terms(self, query_words: List[str]) -> List[Dict[str, Any]]:
        """Match terms based on keywords"""
        matches = []
        
        if not self.keywords_index:
            return matches
          # Score terms based on keyword matches
        term_scores = defaultdict(int)
        
        for word in query_words:
            word_lower = word.lower()
            if word_lower in self.keywords_index:
                for term in self.keywords_index[word_lower]:
                    term_scores[term] += 1
        
        # Add high-scoring terms (threshold: at least 2 matches)
        for term, score in term_scores.items():
            if score >= 2:
                definition = self.term_definitions.get(term.lower(), "No definition available")
                matches.append({
                    'term': term,
                    'definition': definition,
                    'confidence': min(0.7, score * 0.2),
                    'match_type': 'keyword',
                    'keyword_score': score,
                    'category': self._get_term_category(term)
                })        
        return matches
    
    def _get_term_category(self, term: str) -> str:
        """Get category for a term"""
        for category, terms in self.terms_by_category.items():
            if any(isinstance(t, dict) and t.get('term', '').lower() == term.lower() for t in terms):
                return category
            elif any(isinstance(t, str) and t.lower() == term.lower() for t in terms):
                return category
        return "General"
    
    def _deduplicate_terms(self, terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate terms, keeping highest confidence"""
        seen_terms = {}
        
        for term_info in terms:
            term_key = term_info['term'].lower()
            if term_key not in seen_terms or term_info['confidence'] > seen_terms[term_key]['confidence']:
                seen_terms[term_key] = term_info
        
        # Sort by confidence (highest first)
        return sorted(seen_terms.values(), key=lambda x: x['confidence'], reverse=True)
    
    def create_enhanced_context(self, query: str, original_context: str) -> str:
        """
        Create enhanced context by adding legal term definitions
        
        Args:
            query: User query
            original_context: Original RAG context
            
        Returns:
            Enhanced context with legal term definitions
        """
        legal_terms = self.extract_legal_terms_from_query(query)
        
        if not legal_terms:
            return original_context
        
        # Create legal terms section
        terms_context = "\n=== LEGAL TERMS DEFINITIONS ===\n"
        
        for i, term_info in enumerate(legal_terms[:8]):  # Limit to top 8 terms
            terms_context += f"{i+1}. **{term_info['term']}** ({term_info.get('category', 'General')})\n"
            terms_context += f"   Definition: {term_info['definition']}\n\n"
        
        terms_context += "\n" + "="*50 + "\n"
        
        # Combine contexts
        enhanced_context = f"{terms_context}\n{original_context}"
        
        logger.info(f"Enhanced context with {len(legal_terms)} legal terms")
        return enhanced_context
    
    def get_related_terms_by_category(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get related terms from same categories as found terms"""
        found_terms = self.extract_legal_terms_from_query(query)
        
        if not found_terms or not self.terms_by_category:
            return []
        
        # Get categories of found terms
        categories = set()
        for term_info in found_terms:
            categories.add(term_info.get('category', 'General'))
        
        # Get related terms from same categories
        related_terms = []
        for category in categories:
            category_terms = self.terms_by_category.get(category, [])
            for term_data in category_terms[:limit]:  # Limit per category
                if isinstance(term_data, dict):
                    related_terms.append({
                        'term': term_data.get('term', ''),
                        'definition': term_data.get('definition', ''),
                        'category': category
                    })
                elif isinstance(term_data, str):
                    definition = self.term_definitions.get(term_data.lower(), "No definition available")
                    related_terms.append({
                        'term': term_data,
                        'definition': definition,
                        'category': category
                    })
        
        return related_terms[:limit]
    
    def create_simplified_explanation_prompt(self, query: str, context: str) -> str:
        """Create a prompt for simplified explanation"""
        return f"""
Context with Legal Terms Definitions:
{context}

User Query: {query}

Please provide a simplified explanation that:
1. Uses the legal terms definitions provided above
2. Explains complex legal concepts in simple language
3. Provides practical guidance
4. Includes relevant examples
5. Avoids legal jargon

Focus on making the law accessible to common people while being accurate.
"""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded legal terms"""
        return {
            'total_terms': len(self.term_definitions),
            'total_variants': len(self.term_variants),
            'categories': list(self.terms_by_category.keys()),
            'average_definition_length': sum(len(d) for d in self.term_definitions.values()) / max(1, len(self.term_definitions)) if self.term_definitions else 0
        }


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
    
    def load_training_data(self) -> List[Document]:
        """Load training data from procedure files and court hierarchy"""
        documents = []
        
        # Define training data directory
        training_dir = Path("training_data")
        
        # Load court hierarchy
        court_hierarchy_file = training_dir / "court_hierarchy.txt"
        if court_hierarchy_file.exists():
            try:
                with open(court_hierarchy_file, 'r', encoding='utf-8') as f:
                    court_content = f.read()
                
                doc = Document(
                    page_content=court_content,
                    metadata={
                        'document_type': 'court_hierarchy',
                        'source_file': str(court_hierarchy_file),
                        'title': 'Indian Court Hierarchy and Structure'
                    }
                )
                documents.append(doc)
                logger.info("Loaded court hierarchy document")
            except Exception as e:
                logger.error(f"Error loading court hierarchy: {e}")
        
        # Load procedure files
        procedure_dir = training_dir / "procedure"
        if procedure_dir.exists():
            for json_file in procedure_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        procedure_data = json.load(f)
                    
                    # Extract key information for better search
                    procedure_type = procedure_data.get('procedure_type', json_file.stem)
                    legal_framework = procedure_data.get('legal_framework', {})
                    
                    # Create comprehensive content
                    content_parts = []
                    content_parts.append(f"Procedure Type: {procedure_type}")
                    
                    if legal_framework:
                        primary_act = legal_framework.get('primary_act', '')
                        if primary_act:
                            content_parts.append(f"Primary Act: {primary_act}")
                        
                        sections = legal_framework.get('reference_sections', legal_framework.get('applicable_sections', []))
                        if sections:
                            content_parts.append(f"Relevant Sections: {', '.join(sections)}")
                      # Add applicable scenarios
                    scenarios = procedure_data.get('applicable_scenarios', [])
                    if scenarios:
                        # Ensure all scenarios are strings
                        scenario_strings = []
                        for scenario in scenarios:
                            if isinstance(scenario, str):
                                scenario_strings.append(scenario)
                            elif isinstance(scenario, dict):
                                scenario_strings.append(scenario.get('description', str(scenario)))
                            else:
                                scenario_strings.append(str(scenario))
                        content_parts.append(f"Applicable Scenarios: {'; '.join(scenario_strings)}")
                      # Add detailed steps
                    steps = procedure_data.get('steps', procedure_data.get('procedure_steps', []))
                    if steps:
                        content_parts.append("Procedure Steps:")
                        for i, step in enumerate(steps, 1):
                            if isinstance(step, dict):
                                # Handle dictionary format (e.g., consumer_complaint_procedure.json)
                                step_title = step.get('title', f'Step {i}')
                                step_desc = step.get('description', '')
                                step_details = step.get('details', [])
                                
                                content_parts.append(f"{i}. {step_title}")
                                if step_desc:
                                    content_parts.append(f"   Description: {step_desc}")
                                if step_details:
                                    if isinstance(step_details, list):
                                        for detail in step_details:
                                            content_parts.append(f"   - {detail}")
                                    else:
                                        content_parts.append(f"   - {step_details}")
                            elif isinstance(step, str):
                                # Handle string format
                                content_parts.append(f"{i}. {step}")
                            else:
                                content_parts.append(f"{i}. {str(step)}")
                      # Add required documents
                    docs = procedure_data.get('required_documents', [])
                    if docs:
                        # Ensure all docs are strings
                        doc_strings = [str(doc) for doc in docs]
                        content_parts.append(f"Required Documents: {'; '.join(doc_strings)}")
                    
                    # Add timeline information
                    timeline = procedure_data.get('timeline', {})
                    if timeline:
                        content_parts.append("Timeline Information:")
                        for key, value in timeline.items():
                            content_parts.append(f"- {key}: {value}")
                    
                    # Add fees information
                    fees = procedure_data.get('fees', {})
                    if fees:
                        content_parts.append("Fee Structure:")
                        for key, value in fees.items():
                            content_parts.append(f"- {key}: {value}")
                    
                    # Add jurisdiction details
                    jurisdiction = procedure_data.get('jurisdiction', {})
                    if jurisdiction:
                        content_parts.append("Jurisdiction:")
                        for key, value in jurisdiction.items():
                            content_parts.append(f"- {key}: {value}")
                      # Add relief available
                    relief = procedure_data.get('relief_available', [])
                    if relief:
                        relief_strings = [str(item) for item in relief]
                        content_parts.append(f"Relief Available: {'; '.join(relief_strings)}")
                    
                    # Add important notes
                    notes = procedure_data.get('important_notes', [])
                    if notes:
                        content_parts.append("Important Notes:")
                        for note in notes:
                            content_parts.append(f"- {note}")
                    
                    # Create document
                    content = "\n\n".join(content_parts)
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            'document_type': 'legal_procedure',
                            'procedure_type': procedure_type,
                            'source_file': str(json_file),
                            'primary_act': legal_framework.get('primary_act', ''),
                            'applicable_scenarios': scenarios,
                            'title': f"{procedure_type} Procedure Guide"
                        }
                    )
                    documents.append(doc)
                    
                except Exception as e:
                    logger.error(f"Error loading procedure file {json_file}: {e}")
        
        logger.info(f"Loaded {len(documents)} training documents")
        return documents

class LegalRAGSystem:
    """RAG system specifically designed for legal queries with case law analysis"""
    
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
        
        # Initialize case law analyzer if available
        self.case_law_analyzer = None
        if CASE_LAW_ANALYZER_AVAILABLE:
            try:
                self.case_law_analyzer = CaseLawAnalyzer()
                logger.info("Case law analyzer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize case law analyzer: {e}")
        
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
        
        # Filter complex metadata to avoid ChromaDB errors
        filtered_docs = filter_complex_metadata(split_docs)
        logger.info(f"Filtered complex metadata from {len(split_docs)} documents")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=filtered_docs,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    
    def create_legal_qa_chain(self):
        """Create QA chain for legal queries using modern LangChain patterns"""
        if not self.retriever:
            logger.error("Retriever not set up. Call setup_vectorstore first.")
            return
        
        try:
            # Create legal-specific prompt template
            legal_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are an AI Legal Assistant specializing in Indian law. Your role is to help common people understand legal concepts in simple, practical terms.

Context from Legal Documents:
{context}

Question: {question}

Instructions:
1. Provide accurate information based on the legal context provided
2. Explain complex legal terms in simple language
3. Use practical examples when helpful
4. Mention relevant legal acts or sections when applicable
5. If you're unsure about something, clearly state your limitations
6. Focus on practical guidance that helps the person understand their situation

Please provide a comprehensive yet easy-to-understand response:"""
            )
            
            # Create QA chain using RetrievalQA (compatible approach)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": legal_prompt},
                return_source_documents=True
            )
            
            logger.info("Legal QA chain created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create QA chain: {e}")
            # Fallback: create a simple chain without custom prompt
            try:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.retriever,
                    return_source_documents=True
                )
                logger.info("Basic QA chain created as fallback")
            except Exception as fallback_error:
                logger.error(f"Even fallback QA chain creation failed: {fallback_error}")
                self.qa_chain = None
    
    async def enhanced_legal_query(self, query: str, include_case_analysis: bool = True) -> Dict[str, Any]:
        """
        Enhanced legal query with case law analysis and strategic guidance
        
        Args:
            query: User's legal question
            include_case_analysis: Whether to include case law analysis
            
        Returns:
            Dictionary with legal response and strategic guidance
        """
        result = {
            'basic_response': '',
            'case_analysis': '',
            'strategic_guidance': '',
            'similar_precedents': '',
            'error': None
        }
        
        try:            # Get basic legal response from RAG
            if self.qa_chain:
                result['basic_response'] = self.qa_chain.invoke({"query": query})
            else:
                result['basic_response'] = "Legal RAG system not initialized properly."
            
            # Add case law analysis if available and requested
            if include_case_analysis and self.case_law_analyzer:
                try:
                    # Get comprehensive case analysis
                    strategy = await self.case_law_analyzer.analyze_legal_situation(query)
                    result['case_analysis'] = self.case_law_analyzer.format_strategy_response(strategy)
                    
                    # Get similar precedents
                    similar_cases = self.case_law_analyzer._find_similar_cases(query, top_k=3)
                    if similar_cases:
                        precedents = []
                        for case in similar_cases:
                            precedents.append(
                                f"**{case.case_name}** ({case.court}, {case.year})\n"
                                f"Relevance: {case.relevance_score:.0%}\n"
                                f"Legal Principle: {case.legal_principle[:200]}...\n"
                            )
                        result['similar_precedents'] = "\n---\n".join(precedents)
                    
                except Exception as e:
                    logger.error(f"Error in case law analysis: {e}")
                    result['error'] = f"Case law analysis failed: {str(e)}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced legal query: {e}")
            result['error'] = str(e)
            return result
        

class LegalAgent:
    """Modern Agentic system for legal assistance using LangGraph"""
    
    def __init__(self, rag_system: LegalRAGSystem):
        self.rag_system = rag_system
        self.graph = None
        self.setup_graph()    
    def legal_search_tool(self, query: str) -> str:
        """Tool for searching legal documents"""
        try:
            result = self.rag_system.qa_chain.invoke({"query": query})
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
        
        # Create comprehensive legal tools using modern @tool decorator        @tool
        def legal_document_search(query: str) -> str:
            """Search through Indian legal acts and regulations to find relevant information.
            
            Args:
                query: The search query about legal matters
                
            Returns:
                Relevant legal information and guidance
            """
            try:
                return self.rag_system.qa_chain.invoke({"query": query})
            except Exception as e:
                return f"Error searching legal documents: {e}"
        
        @tool
        def case_law_analysis(query: str) -> str:
            """Analyze legal situations using case law precedents and provide strategic guidance.
            
            Args:
                query: The legal situation or question to analyze
                
            Returns:
                Comprehensive legal strategy with precedents and recommendations
            """
            return self.case_law_analysis_tool(query)
        
        @tool
        def find_similar_precedents(query: str) -> str:
            """Find similar legal precedents from case law database.
            
            Args:
                query: Legal query or situation
                
            Returns:
                List of similar legal precedents with relevance scores
            """
            return self.find_precedents_tool(query)
        
        @tool
        def strategic_legal_guidance(situation: str) -> str:
            """Provide strategic legal guidance for a given situation.
            
            Args:
                situation: Description of the legal situation
                
            Returns:
                Strategic advice including next steps and recommendations
            """
            return self.strategic_guidance_tool(situation)
        
        # Get the latest message
        messages = state["messages"]
        
        # Available tools list
        available_tools = [legal_document_search, find_similar_precedents, strategic_legal_guidance]
        
        # Add case law analysis if available
        if self.rag_system.case_law_analyzer:
            available_tools.append(case_law_analysis)
        
        # Use LLM to decide if tools are needed
        response = self.rag_system.llm.bind_tools(available_tools).invoke(messages)
        
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
    """Main interface for the legal assistant with integrated legal terms support and live case law search"""
    
    def __init__(self):
        self.processor = LegalDataProcessor()
        self.rag_system = LegalRAGSystem()
        self.agent = None
        self.is_initialized = False
        # Initialize legal terms integrator
        project_root = Path(__file__).parent.parent
        self.legal_terms_integrator = LegalTermsIntegrator(str(project_root / "legal_terms"))
        # Initialize Indian Kanoon integration - Using Clean Production Client
        self.indian_kanoon_client = None
        if INDIAN_KANOON_AVAILABLE:
            try:
                # Try to get API token from environment
                api_token = os.getenv('INDIAN_KANOON_API_TOKEN')
                
                if api_token:
                    self.indian_kanoon_client = create_indian_kanoon_client(
                        api_token=api_token,
                        budget_limit=500.0
                    )
                    logger.info("Indian Kanoon integration initialized successfully")
                else:
                    logger.warning("Indian Kanoon API token not found in environment")
                    self.indian_kanoon_client = None
                    
            except Exception as e:
                logger.warning(f"Failed to initialize Indian Kanoon: {e}")
                self.indian_kanoon_client = None
    
    def initialize(self):
        """Initialize the system with legal acts and training data"""
        try:
            logger.info("Initializing Legal Assistant...")
            
            # Load legal documents (acts)
            documents = self.processor.load_legal_acts()
            logger.info(f"Loaded {len(documents)} legal act documents")
            
            # Load training data (procedures, court hierarchy)
            training_documents = self.processor.load_training_data()
            logger.info(f"Loaded {len(training_documents)} training documents")
            
            # Combine all documents
            all_documents = documents + training_documents
            
            if not all_documents:
                logger.error("No legal documents found!")
                return False
                
            logger.info(f"Total documents for RAG system: {len(all_documents)}")
            
            # Setup RAG system with all documents
            self.rag_system.setup_vectorstore(all_documents)
            self.rag_system.create_legal_qa_chain()
            
            # Setup agent
            self.agent = LegalAgent(self.rag_system)
            
            # Verify legal terms integration
            stats = self.legal_terms_integrator.get_statistics()
            logger.info(f"Legal terms loaded: {stats['total_terms']} terms")
            
            self.is_initialized = True
            logger.info("Legal Assistant initialized successfully with comprehensive training data!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    def ask_question(self, question: str, context: Optional[str] = None, include_term_analysis: bool = True) -> Dict[str, Any]:
        """Ask a legal question with enhanced legal terms context"""
        if not self.is_initialized:
            return {
                "error": "System not initialized. Please call initialize() first.",
                "question": question,
                "response": "System not initialized. Please call initialize() first.",
                "legal_terms": [],
                "status": "error"
            }
        
        try:
            logger.info(f"Processing question: {question}")
            
            # Step 1: Extract legal terms from the question
            legal_terms = []
            enhanced_context = ""
            
            if include_term_analysis:
                legal_terms = self.legal_terms_integrator.extract_legal_terms_from_query(question)
                logger.info(f"Found {len(legal_terms)} legal terms in question")
            
            # Prepare the query
            if context:
                full_query = f"Context: {context}\n\nQuestion: {question}"
            else:
                full_query = question
            
            # Step 2: Get base response from agent/RAG
            try:
                # Try agent first
                base_response = self.agent.invoke({"input": full_query})
                base_answer = base_response.get("output", "No response generated")
            except Exception as agent_error:
                logger.warning(f"Agent failed, falling back to RAG: {agent_error}")
                # Fallback to direct RAG
                rag_result = self.rag_system.qa_chain.invoke({"query": full_query})
                base_answer = rag_result.get("result", "No response generated")
            
            # Step 3: Create enhanced context with legal terms
            if legal_terms:
                enhanced_context = self.legal_terms_integrator.create_enhanced_context(
                    question, str(base_answer)
                )
                
                # Step 4: Get enhanced response with legal terms context
                logger.info("Getting enhanced response with legal terms...")
                enhanced_prompt = self.legal_terms_integrator.create_simplified_explanation_prompt(
                    question, enhanced_context
                )
                
                try:
                    enhanced_result = self.rag_system.qa_chain.invoke({"query": enhanced_prompt})
                    final_response = enhanced_result.get("result", str(base_answer))
                except Exception:
                    # Use base response if enhancement fails
                    final_response = str(base_answer)
            else:
                # Use base response if no legal terms found
                final_response = str(base_answer)
            
            # Step 5: Get related terms by category for additional context
            related_terms = self.legal_terms_integrator.get_related_terms_by_category(question, limit=3)
            
            return {
                "question": question,
                "response": final_response,
                "legal_terms": legal_terms,
                "related_terms": related_terms,
                "enhanced_context": enhanced_context,
                "base_response": str(base_answer),
                "terms_count": len(legal_terms),
                "context": context,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            # Enhanced fallback with legal terms context
            try:
                logger.info("Attempting enhanced fallback...")
                
                # Extract legal terms even in fallback
                legal_terms = []
                if include_term_analysis:
                    legal_terms = self.legal_terms_integrator.extract_legal_terms_from_query(question)
                
                # Direct RAG query with enhanced context if terms found
                if legal_terms:
                    enhanced_context = self.legal_terms_integrator.create_enhanced_context(question, "")
                    enhanced_prompt = self.legal_terms_integrator.create_simplified_explanation_prompt(question, enhanced_context)
                    rag_result = self.rag_system.qa_chain.invoke({"query": enhanced_prompt})
                else:
                    rag_result = self.rag_system.qa_chain.invoke({"query": question})
                
                return {
                    "question": question,
                    "response": rag_result.get("result", "No response generated"),
                    "legal_terms": legal_terms,
                    "related_terms": [],
                    "context": context,
                    "status": "success_fallback",
                    "source_documents": rag_result.get("source_documents", [])
                }
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return {
                    "question": question,
                    "error": f"Primary error: {str(e)}, Fallback error: {str(fallback_error)}",
                    "legal_terms": [],
                    "status": "error"
                }
    
    def explain_legal_term(self, term: str) -> Dict[str, Any]:
        """
        Get detailed explanation of a specific legal term
        
        Args:
            term: Legal term to explain
            
        Returns:
            Detailed explanation of the term
        """
        try:
            # Get definition from legal terms
            definition = self.legal_terms_integrator.term_definitions.get(term.lower())
            
            if not definition:
                # Try variants
                original_term = self.legal_terms_integrator.term_variants.get(term.lower())
                if original_term:
                    definition = self.legal_terms_integrator.term_definitions.get(original_term.lower())
                    term = original_term
            
            if definition:
                # Get related terms
                related_terms = self.legal_terms_integrator.get_related_terms_by_category(term, limit=5)
                
                # Create enhanced explanation using RAG
                explanation_query = f"Please explain the legal term '{term}' in simple language with examples and practical implications."
                enhanced_context = f"Legal Term Definition: {term} - {definition}"
                
                try:
                    detailed_result = self.rag_system.qa_chain.invoke({
                        "query": f"Context: {enhanced_context}\n\nQuery: {explanation_query}"
                    })
                    detailed_explanation = detailed_result.get("result", definition)
                except Exception:
                    detailed_explanation = definition
                
                return {
                    'term': term,
                    'definition': definition,
                    'detailed_explanation': str(detailed_explanation),
                    'related_terms': related_terms,
                    'found': True
                }
            else:
                return {
                    'term': term,
                    'definition': f"I don't have information about the term '{term}' in my legal terms database.",
                    'detailed_explanation': '',
                    'related_terms': [],
                    'found': False
                }
                
        except Exception as e:
            logger.error(f"Error explaining term '{term}': {str(e)}")
            return {
                'term': term,
                'definition': f"Error retrieving information for term '{term}'",
                'detailed_explanation': '',
                'related_terms': [],
                'found': False,
                'error': str(e)
            }
    
    def analyze_legal_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze a legal text for legal terms and provide explanations
        
        Args:
            text: Legal text to analyze
            
        Returns:
            Analysis with detected terms and explanations
        """
        try:
            # Extract legal terms
            legal_terms = self.legal_terms_integrator.extract_legal_terms_from_query(text)
            
            # Categorize terms
            categories = set()
            for term_info in legal_terms:
                categories.add(term_info.get('category', 'General'))
            
            # Create summary analysis
            analysis_query = f"Analyze this legal text and explain the key legal concepts: {text}"
            enhanced_context = self.legal_terms_integrator.create_enhanced_context(analysis_query, "")
            
            try:
                analysis_result = self.rag_system.qa_chain.invoke({
                    "query": f"Context: {enhanced_context}\n\nQuery: {analysis_query}"
                })
                legal_analysis = analysis_result.get("result", "Analysis not available")
            except Exception:
                legal_analysis = f"Found {len(legal_terms)} legal terms in the text."
            
            return {
                'text': text,
                'detected_terms': legal_terms,
                'total_terms_found': len(legal_terms),
                'categories': list(categories),
                'legal_analysis': str(legal_analysis),
                'summary': {
                    'main_legal_areas': list(categories),
                    'key_terms_count': len(legal_terms),
                    'complexity_level': 'High' if len(legal_terms) > 5 else 'Medium' if len(legal_terms) > 2 else 'Low'
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing legal text: {str(e)}")
            return {
                'text': text,
                'detected_terms': [],
                'total_terms_found': 0,
                'categories': [],
                'legal_analysis': f"Error analyzing text: {str(e)}",
                'summary': {},
                'error': str(e)
            }
    
    def get_terms_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all legal terms in a specific category"""
        return self.legal_terms_integrator.terms_by_category.get(category, [])
    
    def get_available_categories(self) -> List[str]:
        """Get list of available legal categories"""
        return list(self.legal_terms_integrator.terms_by_category.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the legal assistant"""
        base_stats = self.legal_terms_integrator.get_statistics()
        
        # Add RAG system stats if available
        rag_stats = {}
        if hasattr(self.rag_system, 'vectorstore') and self.rag_system.vectorstore:
            try:
                # Try to get vector store stats
                rag_stats['documents_in_vectorstore'] = len(self.rag_system.vectorstore.get()['ids'])
            except:
                rag_stats['documents_in_vectorstore'] = 'Unknown'
        
        return {
            'legal_terms': base_stats,
            'rag_system': rag_stats,
            'initialized': self.is_initialized
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
    
    def search_case_law(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search for relevant case law using Indian Kanoon API.
        
        Args:
            query: Legal query to search for
            max_results: Maximum number of case law results to return
            
        Returns:
            Dictionary containing case law results and metadata
        """
        if not self.indian_kanoon_client:
            return {
                'query': query,
                'results': [],
                'status': 'unavailable',
                'message': 'Indian Kanoon API not available. Please check your configuration.'
            }
        
        try:
            logger.info(f"Searching case law for: {query}")
            
            # Search using the new client
            search_results = self.indian_kanoon_client.search(
                query=query,
                max_results=max_results
            )
            
            # Convert to our format
            case_law_results = []
            for result in search_results:
                case_law_results.append({
                    'doc_id': result.doc_id,
                    'title': result.title,
                    'court': result.court,
                    'date': result.date,
                    'snippet': result.snippet if result.snippet else 'No preview available',
                    'relevance_position': result.position
                })
            
            # Get budget status
            budget_status = self.indian_kanoon_client.get_budget_status()
            
            return {
                'query': query,
                'results': case_law_results,
                'status': 'success',
                'count': len(case_law_results),
                'budget_used': budget_status['current_budget'],
                'budget_remaining': budget_status['remaining_budget'],
                'message': f'Found {len(case_law_results)} relevant case law results'
            }
            
        except Exception as e:
            logger.error(f"Case law search failed: {e}")
            return {
                'query': query,
                'results': [],
                'status': 'error',
                'message': f'Error searching case law: {str(e)}'
            }
    
    def get_case_law_document(self, doc_id: int) -> Dict[str, Any]:
        """
        Retrieve full case law document by ID.
        
        Args:
            doc_id: Indian Kanoon document ID
            
        Returns:
            Dictionary containing full document details
        """
        if not self.indian_kanoon_client:
            return {
                'doc_id': doc_id,
                'status': 'unavailable',
                'message': 'Indian Kanoon API not available'
            }
        
        try:
            logger.info(f"Fetching case law document: {doc_id}")
            
            # Fetch document using the wrapper
            document = self.indian_kanoon_client.get_document(doc_id, include_citations=True)
            
            # Get budget status
            budget_status = self.indian_kanoon_client.get_budget_status()
            
            return {
                'doc_id': doc_id,
                'title': document.title,
                'content': document.content,
                'court': document.court,
                'date': document.date,
                'citations': document.citations,
                'cited_by': document.cited_by,
                'status': 'success',
                'budget_used': budget_status['current_budget'],
                'budget_remaining': budget_status['remaining_budget']
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch document {doc_id}: {e}")
            return {
                'doc_id': doc_id,
                'status': 'error',
                'message': f'Error fetching document: {str(e)}'
            }
    
    def ask_question_with_case_law(self, question: str, context: Optional[str] = None, 
                                   include_case_law: bool = True, max_case_results: int = 3) -> Dict[str, Any]:
        """
        Ask a legal question with optional case law integration.
        
        Args:
            question: Legal question to ask
            context: Additional context
            include_case_law: Whether to search for relevant case law
            max_case_results: Maximum case law results to include
            
        Returns:
            Enhanced response with case law if available
        """
        # Get base response
        base_response = self.ask_question(question, context, include_term_analysis=True)
        
        # Add case law if requested and available
        if include_case_law and self.indian_kanoon_client:
            try:
                logger.info("Adding case law to response...")
                
                # Create a focused query for case law search
                case_law_query = self._create_case_law_query(question, base_response)
                
                # Search for case law
                case_law_results = self.search_case_law(case_law_query, max_case_results)
                
                # Add to response
                base_response['case_law'] = case_law_results
                base_response['enhanced_with_case_law'] = True
                
                # Update response with case law context if found
                if case_law_results['results']:
                    case_law_context = self._format_case_law_for_context(case_law_results['results'])
                    enhanced_prompt = f"""
                    Original Question: {question}
                    
                    Legal Context: {base_response.get('response', '')}
                    
                    Relevant Case Law:
                    {case_law_context}
                    
                    Please provide a comprehensive answer that integrates the legal context with the relevant case law findings.
                    """
                    
                    try:
                        enhanced_result = self.rag_system.qa_chain.invoke({"query": enhanced_prompt})
                        base_response['enhanced_response'] = enhanced_result.get("result", base_response['response'])
                    except Exception as e:
                        logger.warning(f"Failed to create enhanced response: {e}")
                        base_response['enhanced_response'] = base_response['response']
                
            except Exception as e:
                logger.warning(f"Failed to add case law to response: {e}")
                base_response['case_law'] = {
                    'status': 'error',
                    'message': f'Case law search failed: {str(e)}'
                }
                base_response['enhanced_with_case_law'] = False
        else:
            base_response['case_law'] = {
                'status': 'skipped',
                'message': 'Case law search not requested or not available'
            }
            base_response['enhanced_with_case_law'] = False
        
        return base_response
    
    def _create_case_law_query(self, question: str, base_response: Dict[str, Any]) -> str:
        """
        Create a focused query for case law search based on the question and base response.
        
        Args:
            question: Original question
            base_response: Base response from RAG system
            
        Returns:
            Optimized query for case law search
        """
        # Extract legal terms from the question
        legal_terms = base_response.get('legal_terms', [])
        
        # Create base query from question
        query_parts = [question]
        
        # Add legal terms if found
        if legal_terms:
            query_parts.extend([term['term'] for term in legal_terms[:3]])  # Top 3 terms
        
        # Join and limit length
        query = ' '.join(query_parts)
          # Limit query length (Indian Kanoon has query length limits)
        if len(query) > 200:
            query = query[:200]
        
        return query
    
    def _format_case_law_for_context(self, case_results: List[Dict[str, Any]]) -> str:
        """
        Format case law results for use in enhanced response generation.
        
        Args:
            case_results: List of case law results
            
        Returns:
            Formatted string for context
        """
        formatted_cases = []
        for case in case_results[:3]:  # Limit to top 3 cases
            formatted_cases.append(f"""
            Case: {case['title']}
            Court: {case['court']}
            Date: {case['date']}
            Summary: {case['snippet']}
            """)
        
        return '\n'.join(formatted_cases)
    
    def get_indian_kanoon_status(self) -> Dict[str, Any]:
        """
        Get status of Indian Kanoon API integration.
        
        Returns:
            Status information including budget and availability
        """
        if not self.indian_kanoon_client:
            return {
                'available': False,
                'status': 'unavailable',
                'message': 'Indian Kanoon API client not initialized'
            }
        
        try:
            budget_status = self.indian_kanoon_client.get_budget_status()
            return {
                'available': True,
                'status': 'active',
                'budget_limit': budget_status['budget_limit'],
                'current_budget': budget_status['current_budget'],
                'remaining_budget': budget_status['remaining_budget'],
                'budget_percentage_used': budget_status['budget_percentage_used'],
                'search_count': budget_status['search_count'],
                'document_count': budget_status['document_count'],
                'message': 'Indian Kanoon API is available and ready'
            }
        except Exception as e:
            return {
                'available': False,
                'status': 'error',
                'message': f'Error getting status: {str(e)}'
            }

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
