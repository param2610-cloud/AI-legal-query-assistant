"""
Enhanced Case Law Analyzer for AI Legal Assistant
================================================

This module provides advanced case law analysis capabilities including:
1. Case law extraction and indexing from training data
2. Verdict and judgment analysis
3. Strategic legal guidance based on precedents
4. Indian Kanoon API integration for similar case search
5. Next steps and strategy recommendations

Key Features:
- Analyzes case facts, legal issues, and judgments
- Extracts legal principles and precedents
- Provides strategic guidance based on similar cases
- Integrates with Indian Kanoon for live case search
- Offers actionable next steps and recommendations
"""

import os
import sys
import json
import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
# Import from correct locations for modern LangChain
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Indian Kanoon API Integration
try:
    from indian_kanoon_client import IndianKanoonClient, create_indian_kanoon_client
    INDIAN_KANOON_AVAILABLE = True
    logger.info("Indian Kanoon API integration loaded successfully")
except ImportError as e:
    logger.warning(f"Indian Kanoon API not available: {e}")
    INDIAN_KANOON_AVAILABLE = False

class CaseCategory(Enum):
    """Case categories for better classification"""
    CONSUMER_PROTECTION = "consumer_protection"
    CRIMINAL_LAW = "criminal_law" 
    FAMILY_LAW = "family_law"
    PROPERTY_LAW = "property_law"
    CIVIL_PROCEDURE = "civil_procedure"
    CHILD_LABOUR = "child_labour"
    DOWRY_PROHIBITION = "dowry_prohibition"
    DRUG_COSMETICS = "drug_cosmetics"
    CONSTITUTIONAL_LAW = "constitutional_law"
    OTHER = "other"

class StrategicAdviceType(Enum):
    """Types of strategic advice"""
    IMMEDIATE_ACTIONS = "immediate_actions"
    LEGAL_REMEDIES = "legal_remedies"
    DOCUMENTATION = "documentation"
    COURT_PROCEDURES = "court_procedures"
    PREVENTIVE_MEASURES = "preventive_measures"
    TIMELINE = "timeline"
    COSTS = "costs"
    RISKS = "risks"

@dataclass
class CaseLawDocument:
    """Structured representation of a case law document"""
    case_id: str
    case_name: str
    court: str
    year: int
    citation: str
    facts: str
    legal_issues: List[str]
    judgment: str
    legal_reasoning: str
    legal_principle: str
    relevant_sections: List[str]
    keywords: List[str]
    category: CaseCategory
    confidence_score: float = 0.0
    relevance_score: float = 0.0
    
    def to_document(self) -> Document:
        """Convert to LangChain Document format with ChromaDB-compatible metadata"""
        content = f"""
Case Name: {self.case_name}
Court: {self.court}
Year: {self.year}
Citation: {self.citation}

Facts: {self.facts}

Legal Issues:
{chr(10).join([f"- {issue}" for issue in self.legal_issues])}

Judgment: {self.judgment}

Legal Reasoning: {self.legal_reasoning}

Legal Principle: {self.legal_principle}

Relevant Sections:
{chr(10).join([f"- {section}" for section in self.relevant_sections])}

Keywords: {", ".join(self.keywords)}
        """
        
        # Convert all metadata to simple scalar types for ChromaDB compatibility
        metadata = {
            'case_id': str(self.case_id) if self.case_id else '',
            'case_name': str(self.case_name) if self.case_name else '',
            'court': str(self.court) if self.court else '',
            'year': int(self.year) if self.year else 0,
            'citation': str(self.citation) if self.citation else '',
            'category': str(self.category.value) if self.category else '',
            'document_type': 'case_law',
            'confidence_score': float(self.confidence_score) if self.confidence_score is not None else 0.0,
            'relevance_score': float(self.relevance_score) if self.relevance_score is not None else 0.0
        }
        
        # Convert list fields to comma-separated strings
        if self.keywords and isinstance(self.keywords, list):
            metadata['keywords'] = ', '.join([str(k) for k in self.keywords])
        else:
            metadata['keywords'] = str(self.keywords) if self.keywords else ''
            
        if self.relevant_sections and isinstance(self.relevant_sections, list):
            metadata['relevant_sections'] = ', '.join([str(s) for s in self.relevant_sections])
        else:
            metadata['relevant_sections'] = str(self.relevant_sections) if self.relevant_sections else ''
        
        return Document(
            page_content=content.strip(),
            metadata=metadata
        )

@dataclass
class LegalStrategy:
    """Strategic legal advice structure"""
    case_analysis: str
    similar_precedents: List[CaseLawDocument]
    immediate_actions: List[str]
    legal_remedies: List[str]
    required_documentation: List[str]
    court_procedures: List[str]
    timeline: Dict[str, str]
    estimated_costs: Dict[str, str]
    risks_and_challenges: List[str]
    preventive_measures: List[str]
    success_probability: str
    alternative_options: List[str]

class CaseLawAnalyzer:
    """
    Advanced case law analyzer for legal strategy and guidance
    """
    
    def __init__(self, case_law_dir: str = "training_data/case_law", 
                 vector_db_path: str = "chroma_db_caselaw"):
        self.case_law_dir = Path(case_law_dir)
        self.vector_db_path = vector_db_path
        self.case_documents: List[CaseLawDocument] = []
        self.vectorstore = None
        self.embeddings = None
        self.llm = None
        self.indian_kanoon_client = None
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_llm()
        self._initialize_vectorstore()
        self._load_case_law_data()
        
        if INDIAN_KANOON_AVAILABLE:
            self._initialize_indian_kanoon()
    
    def _initialize_embeddings(self):
        """Initialize embedding model with modern HuggingFace integration"""
        try:
            # Try modern langchain-huggingface first
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
            except ImportError:
                # Fallback to community version
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'}
                    )
                except ImportError:
                    # Final fallback to legacy version (with deprecation warning)
                    from langchain.embeddings import HuggingFaceEmbeddings
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'}
                    )
            logger.info("Embeddings model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _initialize_llm(self):
        """Initialize local LLM"""
        try:
            self.llm = ChatOllama(
                model="llama3.2:3b",
                temperature=0.1,
                top_p=0.9
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _initialize_vectorstore(self):
        """Initialize vector database with modern Chroma integration"""
        try:
            # Use community version since langchain-chroma is not commonly available
            from langchain_community.vectorstores import Chroma
            self.vectorstore = Chroma(
                collection_name="case_law_collection",
                embedding_function=self.embeddings,
                persist_directory=self.vector_db_path
            )
            logger.info("Vector database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    def _initialize_indian_kanoon(self):
        """Initialize Indian Kanoon client"""
        try:
            # You would need to provide your API token here
            api_token = os.getenv("INDIAN_KANOON_API_TOKEN")
            if api_token:
                self.indian_kanoon_client = create_indian_kanoon_client(
                    api_token=api_token,
                    budget_limit=100.0
                )
                logger.info("Indian Kanoon client initialized")
            else:
                logger.warning("Indian Kanoon API token not found in environment")
        except Exception as e:
            logger.error(f"Failed to initialize Indian Kanoon client: {e}")
    
    def _load_case_law_data(self):
        """Load and process case law data from JSON files"""
        logger.info(f"Loading case law data from {self.case_law_dir}")
        
        if not self.case_law_dir.exists():
            logger.error(f"Case law directory not found: {self.case_law_dir}")
            return
        
        for json_file in self.case_law_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Determine category from filename
                category = self._determine_category(json_file.stem)
                
                # Handle different JSON structures
                cases = []
                if isinstance(data, list):
                    cases = data
                elif isinstance(data, dict):
                    if 'cases' in data:
                        cases = data['cases']
                    else:
                        cases = [data]  # Single case object
                
                for case_data in cases:
                    if self._is_valid_case_data(case_data):
                        case_doc = self._parse_case_data(case_data, category)
                        if case_doc:
                            self.case_documents.append(case_doc)
                
                logger.info(f"Loaded {len(cases)} cases from {json_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue
        
        logger.info(f"Total case law documents loaded: {len(self.case_documents)}")
        
        # Index documents in vector database
        self._index_case_documents()
    
    def _determine_category(self, filename: str) -> CaseCategory:
        """Determine case category from filename"""
        filename_lower = filename.lower()
        
        category_mapping = {
            'consumer': CaseCategory.CONSUMER_PROTECTION,
            'criminal': CaseCategory.CRIMINAL_LAW,
            'family': CaseCategory.FAMILY_LAW,
            'marriage': CaseCategory.FAMILY_LAW,
            'birth_death': CaseCategory.FAMILY_LAW,
            'property': CaseCategory.PROPERTY_LAW,
            'civil': CaseCategory.CIVIL_PROCEDURE,
            'child_labour': CaseCategory.CHILD_LABOUR,
            'dowry': CaseCategory.DOWRY_PROHIBITION,
            'drug': CaseCategory.DRUG_COSMETICS,
            'cosmetics': CaseCategory.DRUG_COSMETICS,
            'adhaar': CaseCategory.CONSTITUTIONAL_LAW,
            'constitutional': CaseCategory.CONSTITUTIONAL_LAW
        }
        
        for key, category in category_mapping.items():
            if key in filename_lower:
                return category
        
        return CaseCategory.OTHER
    
    def _is_valid_case_data(self, case_data: Dict) -> bool:
        """Check if case data has minimum required fields"""
        required_fields = ['case_name', 'facts', 'judgment']
        return all(field in case_data and case_data[field] for field in required_fields)
    
    def _parse_case_data(self, case_data: Dict, category: CaseCategory) -> Optional[CaseLawDocument]:
        """Parse case data into structured format"""
        try:
            return CaseLawDocument(
                case_id=case_data.get('case_id', ''),
                case_name=case_data.get('case_name', ''),
                court=case_data.get('court', ''),
                year=int(case_data.get('year', 0)),
                citation=case_data.get('citation', ''),
                facts=case_data.get('facts', ''),
                legal_issues=case_data.get('legal_issues', []),
                judgment=case_data.get('judgment', ''),
                legal_reasoning=case_data.get('legal_reasoning', ''),
                legal_principle=case_data.get('legal_principle', ''),                relevant_sections=case_data.get('relevant_sections', []),
                keywords=case_data.get('keywords', []),
                category=category
            )
        except Exception as e:
            logger.error(f"Error parsing case data: {e}")
            return None
    
    def _index_case_documents(self):
        """Index case documents in vector database"""
        if not self.case_documents:
            logger.warning("No case documents to index")
            return
        
        logger.info("Indexing case documents in vector database...")
        
        documents = [case_doc.to_document() for case_doc in self.case_documents]
        
        try:
            # Filter complex metadata as an additional safety measure
            from langchain_community.vectorstores.utils import filter_complex_metadata
            filtered_documents = filter_complex_metadata(documents)
            
            # Check if collection already has documents
            existing_count = self.vectorstore._collection.count()
            
            if existing_count == 0:
                # Add all documents
                self.vectorstore.add_documents(filtered_documents)
                logger.info(f"Indexed {len(filtered_documents)} case law documents")
            else:
                logger.info(f"Vector database already contains {existing_count} documents")
                # Optionally update with new documents
                self.vectorstore.add_documents(filtered_documents)
                logger.info(f"Added {len(filtered_documents)} additional documents")
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
    
    async def analyze_legal_situation(self, user_query: str, context: str = "") -> LegalStrategy:
        """
        Analyze user's legal situation and provide comprehensive strategy
        
        Args:
            user_query: User's legal question or situation
            context: Additional context about the situation
            
        Returns:
            LegalStrategy object with comprehensive guidance
        """
        logger.info(f"Analyzing legal situation: {user_query[:100]}...")
        
        # Step 1: Find similar cases from local database
        similar_cases = self._find_similar_cases(user_query, top_k=5)
        
        # Step 2: If no relevant local cases, search Indian Kanoon
        if not similar_cases or max(case.relevance_score for case in similar_cases) < 0.6:
            logger.info("Searching Indian Kanoon for additional cases...")
            online_cases = await self._search_indian_kanoon(user_query)
            similar_cases.extend(online_cases)
        
        # Step 3: Analyze the situation using LLM
        case_analysis = await self._generate_case_analysis(user_query, context, similar_cases)
        
        # Step 4: Generate strategic recommendations
        strategy = await self._generate_legal_strategy(user_query, context, similar_cases, case_analysis)
        
        return strategy
    
    def _find_similar_cases(self, query: str, top_k: int = 5) -> List[CaseLawDocument]:
        """Find similar cases from local vector database"""
        try:
            # Search vector database
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            
            similar_cases = []
            for doc, score in results:
                # Find corresponding case document
                case_id = doc.metadata.get('case_id', '')
                case_doc = next((case for case in self.case_documents if case.case_id == case_id), None)
                
                if case_doc:
                    case_doc.relevance_score = 1.0 - score  # Convert distance to similarity
                    similar_cases.append(case_doc)
            
            logger.info(f"Found {len(similar_cases)} similar cases in local database")
            return similar_cases
            
        except Exception as e:
            logger.error(f"Error finding similar cases: {e}")
            return []
    
    async def _search_indian_kanoon(self, query: str) -> List[CaseLawDocument]:
        """Search Indian Kanoon for additional relevant cases"""
        if not self.indian_kanoon_client:
            logger.warning("Indian Kanoon client not available")
            return []
        
        try:
            # Use agentic search for better results
            search_result = await self.indian_kanoon_client.agentic_search(query)
            
            online_cases = []
            if search_result.get('status') == 'success':
                cases = search_result.get('cases', [])
                
                for case_data in cases[:3]:  # Limit to top 3 online cases
                    # Convert Indian Kanoon case to our format
                    case_doc = self._convert_ik_case_to_document(case_data)
                    if case_doc:
                        online_cases.append(case_doc)
                
                logger.info(f"Found {len(online_cases)} relevant cases from Indian Kanoon")
            
            return online_cases
            
        except Exception as e:
            logger.error(f"Error searching Indian Kanoon: {e}")
            return []
    
    def _convert_ik_case_to_document(self, case_data: Dict) -> Optional[CaseLawDocument]:
        """Convert Indian Kanoon case data to our format"""
        try:
            return CaseLawDocument(
                case_id=case_data.get('tid', ''),
                case_name=case_data.get('title', ''),
                court=case_data.get('court', ''),
                year=int(case_data.get('judgmentdate', '0')[:4]) if case_data.get('judgmentdate') else 0,
                citation=case_data.get('citation', ''),
                facts=case_data.get('summary', ''),
                legal_issues=[],
                judgment=case_data.get('summary', ''),
                legal_reasoning='',
                legal_principle='',
                relevant_sections=[],
                keywords=case_data.get('keywords', []),
                category=CaseCategory.OTHER,
                relevance_score=case_data.get('relevance_score', 0.7)
            )
        except Exception as e:
            logger.error(f"Error converting Indian Kanoon case: {e}")
            return None
    
    async def _generate_case_analysis(self, query: str, context: str, 
                                    similar_cases: List[CaseLawDocument]) -> str:
        """Generate comprehensive case analysis using LLM"""
        
        # Prepare precedent information
        precedents_text = ""
        if similar_cases:
            precedents_text = "\n\n".join([
                f"**{case.case_name}** ({case.court}, {case.year})\n"
                f"Facts: {case.facts[:200]}...\n"
                f"Legal Principle: {case.legal_principle}\n"
                f"Relevance Score: {case.relevance_score:.2f}"
                for case in similar_cases[:3]
            ])
        
        analysis_prompt = f"""
You are an expert Indian legal advisor. Analyze the following legal situation and provide a comprehensive analysis.

**User's Situation:**
{query}

**Additional Context:**
{context}

**Relevant Legal Precedents:**
{precedents_text}

**Analysis Requirements:**
1. Identify the key legal issues involved
2. Assess the strength of the case based on precedents
3. Identify applicable laws and legal provisions
4. Highlight potential challenges and opportunities
5. Provide preliminary assessment of likely outcomes

Please provide a detailed legal analysis in simple, understandable language.
        """
        
        try:
            # Try async first
            response = await self.llm.ainvoke(analysis_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error generating case analysis: {e}")
            # Fallback to synchronous call
            try:
                response = self.llm.invoke(analysis_prompt)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e2:
                logger.error(f"Error with sync fallback: {e2}")
                return self._generate_fallback_analysis(query, context, similar_cases)
    
    async def _generate_legal_strategy(self, query: str, context: str, 
                                     similar_cases: List[CaseLawDocument],
                                     case_analysis: str) -> LegalStrategy:
        """Generate comprehensive legal strategy"""
        
        strategy_prompt = f"""
Based on the legal analysis and precedents, provide a comprehensive legal strategy and action plan.

**User's Situation:** {query}
**Context:** {context}
**Legal Analysis:** {case_analysis}

Please provide a detailed strategy covering:

1. **Immediate Actions** (What to do right now)
2. **Legal Remedies** (Available legal options)
3. **Required Documentation** (Documents needed)
4. **Court Procedures** (If litigation is needed)
5. **Timeline** (Expected duration for each step)
6. **Estimated Costs** (Approximate expenses)
7. **Risks and Challenges** (Potential problems)
8. **Preventive Measures** (How to avoid future issues)
9. **Success Probability** (Realistic assessment)
10. **Alternative Options** (Other approaches)

Format your response as JSON with these sections.
"""
        
        try:
            # Try async first
            response = await self.llm.ainvoke(strategy_prompt)
            strategy_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse LLM response and create strategy object
            strategy = self._parse_strategy_response(strategy_text, similar_cases)
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating legal strategy: {e}")
            # Fallback to synchronous call
            try:
                response = self.llm.invoke(strategy_prompt)
                strategy_text = response.content if hasattr(response, 'content') else str(response)
                strategy = self._parse_strategy_response(strategy_text, similar_cases)
                return strategy
            except Exception as e2:
                logger.error(f"Error with sync fallback: {e2}")
                return self._create_fallback_strategy(similar_cases)
    
    def _parse_strategy_response(self, strategy_text: str, 
                               similar_cases: List[CaseLawDocument]) -> LegalStrategy:
        """Parse LLM strategy response into structured format"""
        
        # Try to extract structured information from the response
        try:
            # Simple parsing based on section headers
            sections = {
                'case_analysis': '',
                'immediate_actions': [],
                'legal_remedies': [],
                'required_documentation': [],
                'court_procedures': [],
                'timeline': {},
                'estimated_costs': {},
                'risks_and_challenges': [],
                'preventive_measures': [],
                'success_probability': '',
                'alternative_options': []
            }
            
            current_section = 'case_analysis'
            lines = strategy_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                line_lower = line.lower()
                if 'immediate action' in line_lower:
                    current_section = 'immediate_actions'
                elif 'legal remed' in line_lower:
                    current_section = 'legal_remedies'
                elif 'documentation' in line_lower or 'document' in line_lower:
                    current_section = 'required_documentation'
                elif 'court procedure' in line_lower:
                    current_section = 'court_procedures'
                elif 'timeline' in line_lower:
                    current_section = 'timeline'
                elif 'cost' in line_lower:
                    current_section = 'estimated_costs'
                elif 'risk' in line_lower or 'challenge' in line_lower:
                    current_section = 'risks_and_challenges'
                elif 'preventive' in line_lower:
                    current_section = 'preventive_measures'
                elif 'success' in line_lower or 'probability' in line_lower:
                    current_section = 'success_probability'
                elif 'alternative' in line_lower:
                    current_section = 'alternative_options'
                else:
                    # Add content to current section
                    if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                        item = line[1:].strip()
                        if current_section in ['immediate_actions', 'legal_remedies', 
                                             'required_documentation', 'court_procedures',
                                             'risks_and_challenges', 'preventive_measures',
                                             'alternative_options']:
                            sections[current_section].append(item)
                    elif ':' in line and current_section in ['timeline', 'estimated_costs']:
                        key, value = line.split(':', 1)
                        sections[current_section][key.strip()] = value.strip()
                    elif current_section == 'success_probability':
                        sections[current_section] += line + ' '
                    elif current_section == 'case_analysis':
                        sections[current_section] += line + '\n'
            
            return LegalStrategy(
                case_analysis=sections['case_analysis'].strip(),
                similar_precedents=similar_cases,
                immediate_actions=sections['immediate_actions'],
                legal_remedies=sections['legal_remedies'],
                required_documentation=sections['required_documentation'],
                court_procedures=sections['court_procedures'],
                timeline=sections['timeline'],
                estimated_costs=sections['estimated_costs'],
                risks_and_challenges=sections['risks_and_challenges'],
                preventive_measures=sections['preventive_measures'],
                success_probability=sections['success_probability'].strip(),
                alternative_options=sections['alternative_options']
            )
            
        except Exception as e:
            logger.error(f"Error parsing strategy response: {e}")
            return self._create_fallback_strategy(similar_cases)
    
    def _create_fallback_strategy(self, similar_cases: List[CaseLawDocument]) -> LegalStrategy:
        """Create fallback strategy when parsing fails"""
        return LegalStrategy(
            case_analysis="Legal analysis could not be generated automatically. Please consult with a legal professional.",
            similar_precedents=similar_cases,
            immediate_actions=["Consult with a qualified lawyer", "Gather all relevant documents"],
            legal_remedies=["To be determined based on specific case details"],
            required_documentation=["All relevant documents related to the case"],
            court_procedures=["To be advised by legal counsel"],
            timeline={"Consultation": "Immediate", "Legal Action": "To be determined"},
            estimated_costs={"Legal Consultation": "₹2,000-5,000", "Court Fees": "Variable"},
            risks_and_challenges=["Legal complexity requires professional guidance"],
            preventive_measures=["Follow legal compliance requirements"],
            success_probability="To be assessed by legal professional",
            alternative_options=["Mediation", "Negotiation", "Alternative Dispute Resolution"]
        )
    
    def format_strategy_response(self, strategy: LegalStrategy) -> str:
        """Format strategy for user-friendly display"""
        
        response = f"""
# 📋 Legal Analysis & Strategy

## 🔍 Case Analysis
{strategy.case_analysis}

## ⚡ Immediate Actions Required
{chr(10).join([f"• {action}" for action in strategy.immediate_actions])}

## ⚖️ Available Legal Remedies
{chr(10).join([f"• {remedy}" for remedy in strategy.legal_remedies])}

## 📄 Required Documentation
{chr(10).join([f"• {doc}" for doc in strategy.required_documentation])}

## 🏛️ Court Procedures (if applicable)
{chr(10).join([f"• {procedure}" for procedure in strategy.court_procedures])}

## ⏰ Expected Timeline
{chr(10).join([f"• **{key}**: {value}" for key, value in strategy.timeline.items()])}

## 💰 Estimated Costs
{chr(10).join([f"• **{key}**: {value}" for key, value in strategy.estimated_costs.items()])}

## ⚠️ Risks & Challenges
{chr(10).join([f"• {risk}" for risk in strategy.risks_and_challenges])}

## 🛡️ Preventive Measures
{chr(10).join([f"• {measure}" for measure in strategy.preventive_measures])}

## 📊 Success Probability
{strategy.success_probability}

## 🔄 Alternative Options
{chr(10).join([f"• {option}" for option in strategy.alternative_options])}

## 📚 Relevant Precedents
{chr(10).join([f"• **{case.case_name}** ({case.court}, {case.year}) - Relevance: {case.relevance_score:.0%}" for case in strategy.similar_precedents[:3]])}

---
*This analysis is based on available case law and should not replace professional legal advice.*
        """
        
        return response.strip()

# Tool functions for integration with existing system
@tool
async def analyze_legal_case(query: str, context: str = "") -> str:
    """
    Analyze a legal situation and provide strategic guidance based on case law precedents.
    
    Args:
        query: The user's legal question or situation
        context: Additional context about the situation
        
    Returns:
        Comprehensive legal strategy and recommendations
    """
    try:
        analyzer = CaseLawAnalyzer()
        strategy = await analyzer.analyze_legal_situation(query, context)
        return analyzer.format_strategy_response(strategy)
    except Exception as e:
        logger.error(f"Error in legal case analysis: {e}")
        return f"Unable to analyze legal case due to technical issues: {str(e)}"

@tool
def find_similar_precedents(query: str, limit: int = 5) -> str:
    """
    Find similar legal precedents from the case law database.
    
    Args:
        query: Legal query or situation
        limit: Maximum number of precedents to return
        
    Returns:
        List of similar legal precedents with relevance scores
    """
    try:
        analyzer = CaseLawAnalyzer()
        similar_cases = analyzer._find_similar_cases(query, top_k=limit)
        
        if not similar_cases:
            return "No similar precedents found in the database."
        
        precedents = []
        for case in similar_cases:
            precedents.append(
                f"**{case.case_name}** ({case.court}, {case.year})\n"
                f"Relevance: {case.relevance_score:.0%}\n"
                f"Legal Principle: {case.legal_principle}\n"
                f"Keywords: {', '.join(case.keywords[:5])}\n"
            )
        
        return "\n---\n".join(precedents)
        
    except Exception as e:
        logger.error(f"Error finding precedents: {e}")
        return f"Unable to find precedents due to technical issues: {str(e)}"

# Example usage and testing
async def test_case_law_analyzer():
    """Test the case law analyzer with sample queries"""
    
    analyzer = CaseLawAnalyzer()
    
    test_queries = [
        "My employer is not paying overtime despite making me work 12 hours daily. What are my rights?",
        "I bought a defective phone online and the seller refuses to refund. What can I do?",
        "My husband is demanding dowry even after marriage. How can I protect myself?",
        "I need to register my child's birth but it's been more than a year. Is it possible?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        strategy = await analyzer.analyze_legal_situation(query)
        formatted_response = analyzer.format_strategy_response(strategy)
        print(formatted_response)
        print("\n")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_case_law_analyzer())
