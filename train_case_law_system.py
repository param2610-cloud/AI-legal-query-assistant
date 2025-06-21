#!/usr/bin/env python3
"""
Enhanced Legal Assistant Integration - Case Law Analysis Training Script
=======================================================================

This script integrates case law analysis capabilities with the existing AI Legal Assistant.
It trains the system on available case law data and provides enhanced legal guidance
with precedent analysis and strategic recommendations.

Features:
1. Case law document processing and indexing
2. Precedent-based legal analysis
3. Strategic guidance generation
4. Indian Kanoon API integration for live case search
5. Enhanced RAG with case law context

Usage:
    python train_case_law_system.py
    python train_case_law_system.py --rebuild-db
    python train_case_law_system.py --test-queries
"""

import os
import sys
import json
import logging
import asyncio
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup the environment and paths"""
    project_root = Path(__file__).parent
    
    # Add directories to Python path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "agent"))
    
    return project_root

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import chromadb
    except ImportError:
        missing_deps.append("chromadb")
    
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        missing_deps.append("langchain-ollama")
    
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        missing_deps.append("langchain-community")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Please install missing dependencies using: pip install -r requirements.txt")
        return False
    
    return True

def load_case_law_data(case_law_dir: Path) -> List[Dict[str, Any]]:
    """Load case law data from JSON files"""
    
    if not case_law_dir.exists():
        logger.error(f"Case law directory not found: {case_law_dir}")
        return []
    
    all_cases = []
    
    for json_file in case_law_dir.glob("*.json"):
        try:
            logger.info(f"Loading cases from {json_file.name}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            cases = []
            if isinstance(data, list):
                cases = data
            elif isinstance(data, dict):
                if 'cases' in data:
                    cases = data['cases']
                else:
                    cases = [data]  # Single case object
            
            # Filter valid cases
            valid_cases = []
            for case in cases:
                if isinstance(case, dict) and case.get('case_name') and case.get('facts'):
                    # Add metadata
                    case['source_file'] = json_file.stem
                    case['case_category'] = determine_case_category(json_file.stem)
                    valid_cases.append(case)
            
            all_cases.extend(valid_cases)
            logger.info(f"Loaded {len(valid_cases)} valid cases from {json_file.name}")
            
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue
    
    logger.info(f"Total case law documents loaded: {len(all_cases)}")
    return all_cases

def determine_case_category(filename: str) -> str:
    """Determine case category from filename"""
    filename_lower = filename.lower()
    
    category_mapping = {
        'consumer': 'Consumer Protection',
        'criminal': 'Criminal Law',
        'family': 'Family Law',
        'marriage': 'Family Law',
        'birth_death': 'Family Law',
        'property': 'Property Law',
        'civil': 'Civil Procedure',
        'child_labour': 'Child Labour',
        'dowry': 'Dowry Prohibition',
        'drug': 'Drug & Cosmetics',
        'cosmetics': 'Drug & Cosmetics',
        'adhaar': 'Constitutional Law',
        'constitutional': 'Constitutional Law'
    }
    
    for key, category in category_mapping.items():
        if key in filename_lower:
            return category
    
    return 'General Law'

def create_case_law_documents(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert case data to document format for vector storage"""
    
    documents = []
    
    for case in cases:
        try:
            # Create comprehensive document content
            content_parts = []
            
            # Basic case information
            content_parts.append(f"Case Name: {case.get('case_name', 'Unknown')}")
            content_parts.append(f"Court: {case.get('court', 'Unknown')}")
            content_parts.append(f"Year: {case.get('year', 'Unknown')}")
            content_parts.append(f"Citation: {case.get('citation', 'Not available')}")
            
            # Case facts
            facts = case.get('facts', '')
            if facts:
                content_parts.append(f"\nFacts:\n{facts}")
            
            # Legal issues
            legal_issues = case.get('legal_issues', [])
            if legal_issues and isinstance(legal_issues, list):
                content_parts.append(f"\nLegal Issues:")
                for issue in legal_issues:
                    content_parts.append(f"- {issue}")
            
            # Judgment
            judgment = case.get('judgment', '')
            if judgment:
                content_parts.append(f"\nJudgment:\n{judgment}")
            
            # Legal reasoning
            legal_reasoning = case.get('legal_reasoning', '')
            if legal_reasoning:
                content_parts.append(f"\nLegal Reasoning:\n{legal_reasoning}")
            
            # Legal principle
            legal_principle = case.get('legal_principle', '')
            if legal_principle:
                content_parts.append(f"\nLegal Principle:\n{legal_principle}")
            
            # Relevant sections
            relevant_sections = case.get('relevant_sections', [])
            if relevant_sections and isinstance(relevant_sections, list):
                content_parts.append(f"\nRelevant Legal Sections:")
                for section in relevant_sections:
                    content_parts.append(f"- {section}")
            
            # Keywords
            keywords = case.get('keywords', [])
            if keywords and isinstance(keywords, list):
                content_parts.append(f"\nKeywords: {', '.join(keywords)}")
            
            # Combine content
            page_content = '\n'.join(content_parts)
            
            # Create document with metadata
            doc = {
                'page_content': page_content,
                'metadata': {
                    'case_id': case.get('case_id', ''),
                    'case_name': case.get('case_name', ''),
                    'court': case.get('court', ''),
                    'year': case.get('year', 0),
                    'citation': case.get('citation', ''),
                    'case_category': case.get('case_category', 'General Law'),
                    'source_file': case.get('source_file', ''),
                    'document_type': 'case_law',
                    'keywords': keywords if isinstance(keywords, list) else [],
                    'relevant_sections': relevant_sections if isinstance(relevant_sections, list) else []
                }
            }
            
            documents.append(doc)
            
        except Exception as e:
            logger.error(f"Error creating document for case {case.get('case_name', 'Unknown')}: {e}")
            continue
    
    logger.info(f"Created {len(documents)} case law documents for indexing")
    return documents

def setup_vector_database(documents: List[Dict[str, Any]], db_path: str = "chroma_db_caselaw"):
    """Setup ChromaDB vector database with case law documents"""
    
    try:
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
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores.utils import filter_complex_metadata
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Convert to LangChain Document format with simple metadata
        langchain_docs = []
        for doc in documents:
            # Simplify metadata to avoid complex data types
            simple_metadata = {}
            for key, value in doc['metadata'].items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    simple_metadata[key] = value
                elif isinstance(value, list):
                    # Convert lists to comma-separated strings
                    if value:  # Only if list is not empty
                        simple_metadata[key] = ', '.join(str(item) for item in value)
                    else:
                        simple_metadata[key] = ''
                else:
                    simple_metadata[key] = str(value)
            
            langchain_doc = Document(
                page_content=doc['page_content'],
                metadata=simple_metadata
            )
            langchain_docs.append(langchain_doc)
        
        # Split documents for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(langchain_docs)
        logger.info(f"Split {len(langchain_docs)} documents into {len(split_docs)} chunks")
        
        # Filter complex metadata to ensure compatibility with ChromaDB
        filtered_docs = filter_complex_metadata(split_docs)
        logger.info(f"Filtered complex metadata from {len(split_docs)} documents")
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=filtered_docs,
            embedding=embeddings,
            persist_directory=db_path
        )
        
        logger.info(f"Vector database created successfully at {db_path}")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error setting up vector database: {e}")
        return None

def create_enhanced_legal_prompt():
    """Create enhanced prompt template for legal analysis with case law"""
    
    return """You are an AI Legal Assistant specializing in Indian law with access to case law precedents. 
Your role is to provide comprehensive legal guidance that includes:

1. Direct answers to legal questions using relevant law and precedents
2. Analysis of similar legal cases and their outcomes
3. Strategic guidance with actionable next steps
4. Risk assessment and success probability
5. Alternative options and preventive measures

Context from Legal Documents and Case Law:
{context}

User Question: {question}

Analysis Framework:
1. **Legal Analysis**: What laws apply and what do they say?
2. **Precedent Analysis**: What do similar cases teach us?
3. **Strategic Guidance**: What should the person do next?
4. **Risk Assessment**: What are the challenges and success probability?
5. **Practical Steps**: Specific actions, documents needed, timeline

Response Guidelines:
- Start with a direct answer
- Reference relevant case law and legal principles
- Provide step-by-step guidance
- Use simple language that common people can understand
- Include warnings about consulting lawyers for complex matters
- Format with clear sections and bullet points

Enhanced Legal Response:"""

class AdvancedCaseLawAnalyzer:
    """Advanced case law analyzer with strategic guidance"""
    
    def __init__(self, vectorstore, llm_model: str = "llama3.2:3b"):
        # Import here to avoid circular imports
        try:
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(model=llm_model, temperature=0.1)
        except ImportError:
            logger.error("ChatOllama not available. Please install langchain-ollama")
            self.llm = None
            
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Get more context for better analysis
        )
        
        # Initialize Indian Kanoon client if available
        self.indian_kanoon_client = None
        try:
            from indian_kanoon_client import create_indian_kanoon_client
            api_token = os.getenv('INDIAN_KANOON_API_TOKEN')
            if api_token:
                try:
                    self.indian_kanoon_client = create_indian_kanoon_client(api_token)
                    logger.info("Indian Kanoon client initialized for live case search")
                except Exception as e:
                    logger.warning(f"Failed to initialize Indian Kanoon client: {e}")
        except ImportError:
            logger.warning("Indian Kanoon client not available")
    
    def analyze_verdict_judgment(self, query: str) -> Dict[str, Any]:
        """Analyze verdicts and judgments for similar cases"""
        
        try:
            # Retrieve relevant cases
            relevant_cases = self.retriever.invoke(query)
            
            analysis = {
                'query': query,
                'total_cases_found': len(relevant_cases),
                'verdict_patterns': [],
                'success_factors': [],
                'failure_factors': [],
                'strategic_insights': [],
                'recommendation': ''
            }
            
            verdicts = []
            success_cases = []
            failure_cases = []
            
            for doc in relevant_cases:
                metadata = doc.metadata
                content = doc.page_content
                
                # Extract verdict information
                verdict_info = self._extract_verdict_info(content, metadata)
                if verdict_info:
                    verdicts.append(verdict_info)
                    
                    # Categorize outcomes
                    if self._is_favorable_outcome(content):
                        success_cases.append(verdict_info)
                    else:
                        failure_cases.append(verdict_info)
            
            # Analyze patterns
            analysis['verdict_patterns'] = self._analyze_verdict_patterns(verdicts)
            analysis['success_factors'] = self._extract_success_factors(success_cases)
            analysis['failure_factors'] = self._extract_failure_factors(failure_cases)
            analysis['strategic_insights'] = self._generate_strategic_insights(verdicts)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing verdicts: {e}")
            return {'error': str(e), 'query': query}
    
    def generate_strategy_recommendations(self, query: str) -> Dict[str, Any]:
        """Generate comprehensive strategy recommendations"""
        
        try:
            # Get verdict analysis
            verdict_analysis = self.analyze_verdict_judgment(query)
            
            # Create strategy prompt
            strategy_prompt = f"""
            Based on the case law analysis for: {query}
            
            Cases analyzed: {verdict_analysis.get('total_cases_found', 0)}
            Verdict patterns: {verdict_analysis.get('verdict_patterns', [])}
            Success factors: {verdict_analysis.get('success_factors', [])}
            Failure factors: {verdict_analysis.get('failure_factors', [])}
            
            Provide comprehensive strategic recommendations including:
            1. Immediate Actions (What to do right now)
            2. Legal Remedies Available (What options exist)
            3. Required Documentation (What evidence/papers needed)
            4. Timeline & Process (Step-by-step timeline)
            5. Success Probability (Realistic assessment)
            6. Risk Mitigation (What to avoid)
            7. Alternative Solutions (Other approaches)
            8. When to Consult Lawyer (Red flags)
            
            Make recommendations practical and actionable for common people.
            """
            
            strategy_response = self.llm.invoke(strategy_prompt)
            
            return {
                'query': query,
                'strategy_recommendation': strategy_response.content,
                'verdict_analysis': verdict_analysis,
                'confidence': self._calculate_confidence(verdict_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error generating strategy: {e}")
            return {'error': str(e), 'query': query}
    
    async def search_similar_cases_online(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for similar cases using Indian Kanoon API"""
        
        if not self.indian_kanoon_client:
            return {
                'status': 'unavailable',
                'message': 'Indian Kanoon API not available',
                'local_results': self._get_local_similar_cases(query, max_results)
            }
        
        try:
            # Use agentic search for better results
            online_results = await self.indian_kanoon_client.agentic_search(query)
            
            # Combine with local analysis
            local_results = self._get_local_similar_cases(query, max_results)
            
            return {
                'status': 'success',
                'online_results': online_results,
                'local_results': local_results,
                'combined_analysis': self._combine_online_local_analysis(online_results, local_results)
            }
            
        except Exception as e:
            logger.error(f"Error searching online cases: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'local_results': self._get_local_similar_cases(query, max_results)
            }
    
    def _extract_verdict_info(self, content: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract verdict information from case content"""
        
        try:
            # Look for judgment section
            judgment_match = re.search(r'Judgment:\s*(.*?)(?:\n\n|\nLegal Reasoning:|$)', content, re.DOTALL)
            if not judgment_match:
                return None
            
            judgment_text = judgment_match.group(1).strip()
            
            # Extract outcome
            outcome = self._determine_outcome(judgment_text)
            
            # Extract key factors
            factors = self._extract_key_factors(content)
            
            return {
                'case_name': metadata.get('case_name', ''),
                'court': metadata.get('court', ''),
                'year': metadata.get('year', ''),
                'judgment': judgment_text,
                'outcome': outcome,
                'key_factors': factors,
                'legal_principle': self._extract_legal_principle(content)
            }
            
        except Exception as e:
            logger.error(f"Error extracting verdict info: {e}")
            return None
    
    def _is_favorable_outcome(self, content: str) -> bool:
        """Determine if the case outcome was favorable to the plaintiff/petitioner"""
        
        favorable_keywords = [
            'granted', 'allowed', 'upheld', 'sustained', 'ruled in favor',
            'petition allowed', 'appeal allowed', 'relief granted', 'compensation awarded',
            'damages awarded', 'injunction granted', 'rights protected'
        ]
        
        unfavorable_keywords = [
            'dismissed', 'rejected', 'denied', 'quashed', 'set aside',
            'petition dismissed', 'appeal dismissed', 'relief denied', 'no relief'
        ]
        
        content_lower = content.lower()
        
        favorable_score = sum(1 for keyword in favorable_keywords if keyword in content_lower)
        unfavorable_score = sum(1 for keyword in unfavorable_keywords if keyword in content_lower)
        
        return favorable_score > unfavorable_score
    
    def _analyze_verdict_patterns(self, verdicts: List[Dict[str, Any]]) -> List[str]:
        """Analyze patterns in verdicts"""
        
        patterns = []
        
        if not verdicts:
            return patterns
        
        # Outcome distribution
        outcomes = [v.get('outcome', 'Unknown') for v in verdicts]
        outcome_counts = {}
        for outcome in outcomes:
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        
        total = len(verdicts)
        for outcome, count in outcome_counts.items():
            percentage = (count / total) * 100
            patterns.append(f"{outcome}: {count}/{total} cases ({percentage:.1f}%)")
        
        # Court patterns
        courts = [v.get('court', 'Unknown') for v in verdicts]
        court_counts = {}
        for court in courts:
            court_counts[court] = court_counts.get(court, 0) + 1
        
        if len(court_counts) > 1:
            most_common_court = max(court_counts.items(), key=lambda x: x[1])
            patterns.append(f"Most cases from: {most_common_court[0]} ({most_common_court[1]} cases)")
        
        return patterns
    
    def _extract_success_factors(self, success_cases: List[Dict[str, Any]]) -> List[str]:
        """Extract factors that led to successful outcomes"""
        
        factors = []
        
        for case in success_cases:
            case_factors = case.get('key_factors', [])
            factors.extend(case_factors)
        
        # Count frequency and return most common factors
        factor_counts = {}
        for factor in factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        # Return top factors
        sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
        return [f"{factor} (mentioned in {count} cases)" for factor, count in sorted_factors[:5]]
    
    def _extract_failure_factors(self, failure_cases: List[Dict[str, Any]]) -> List[str]:
        """Extract factors that led to unsuccessful outcomes"""
        
        factors = []
        
        for case in failure_cases:
            case_factors = case.get('key_factors', [])
            factors.extend(case_factors)
        
        # Count frequency and return most common factors
        factor_counts = {}
        for factor in factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        # Return top factors
        sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
        return [f"{factor} (mentioned in {count} cases)" for factor, count in sorted_factors[:5]]
    
    def _generate_strategic_insights(self, verdicts: List[Dict[str, Any]]) -> List[str]:
        """Generate strategic insights from verdict analysis"""
        
        insights = []
        
        if not verdicts:
            return insights
        
        # Timeline analysis
        years = [v.get('year', 0) for v in verdicts if v.get('year', 0) > 0]
        if years:
            recent_cases = [y for y in years if y >= 2020]
            if recent_cases:
                insights.append(f"Recent trends: {len(recent_cases)} similar cases in last 5 years")
        
        # Legal principle analysis
        principles = [v.get('legal_principle', '') for v in verdicts if v.get('legal_principle')]
        if principles:
            insights.append(f"Key legal principles established in {len(principles)} cases")
        
        # Court hierarchy insights
        supreme_court_cases = [v for v in verdicts if 'supreme court' in v.get('court', '').lower()]
        high_court_cases = [v for v in verdicts if 'high court' in v.get('court', '').lower()]
        
        if supreme_court_cases:
            insights.append(f"Supreme Court precedents: {len(supreme_court_cases)} cases")
        if high_court_cases:
            insights.append(f"High Court decisions: {len(high_court_cases)} cases")
        
        return insights
    
    def _determine_outcome(self, judgment_text: str) -> str:
        """Determine the outcome of a case from judgment text"""
        
        judgment_lower = judgment_text.lower()
        
        if any(word in judgment_lower for word in ['granted', 'allowed', 'upheld', 'sustained']):
            return 'Favorable'
        elif any(word in judgment_lower for word in ['dismissed', 'rejected', 'denied', 'quashed']):
            return 'Unfavorable'
        elif any(word in judgment_lower for word in ['remanded', 'sent back', 'further inquiry']):
            return 'Remanded'
        elif any(word in judgment_lower for word in ['settled', 'compromise', 'mutual agreement']):
            return 'Settled'
        else:
            return 'Mixed/Unclear'
    
    def _extract_key_factors(self, content: str) -> List[str]:
        """Extract key factors from case content"""
        
        factors = []
        
        # Look for common legal factors
        factor_patterns = {
            'evidence': r'evidence\s+(?:was|is|shows|indicates|proves)',
            'procedure': r'(?:proper|improper|correct|incorrect)\s+procedure',
            'documentation': r'(?:adequate|inadequate|proper|improper)\s+(?:documentation|documents)',
            'time_limit': r'(?:within|beyond|after)\s+(?:time|prescribed|statutory)\s+(?:limit|period)',
            'jurisdiction': r'(?:proper|improper|correct|incorrect)\s+jurisdiction',
            'compliance': r'(?:compliance|non-compliance)\s+with'
        }
        
        content_lower = content.lower()
        
        for factor_type, pattern in factor_patterns.items():
            if re.search(pattern, content_lower):
                factors.append(factor_type.replace('_', ' ').title())
        
        return factors
    
    def _extract_legal_principle(self, content: str) -> str:
        """Extract legal principle from case content"""
        
        # Look for legal principle section
        principle_match = re.search(r'Legal Principle:\s*(.*?)(?:\n\n|\nRelevant|$)', content, re.DOTALL)
        if principle_match:
            return principle_match.group(1).strip()
        
        # Look for legal reasoning section
        reasoning_match = re.search(r'Legal Reasoning:\s*(.*?)(?:\n\n|\nLegal Principle:|$)', content, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            # Extract first sentence as principle
            sentences = reasoning.split('.')
            if sentences:
                return sentences[0].strip() + '.'
        
        return ''
    
    def _get_local_similar_cases(self, query: str, max_results: int) -> Dict[str, Any]:
        """Get similar cases from local database"""
        
        try:
            docs = self.retriever.invoke(query)[:max_results]
            
            similar_cases = []
            for doc in docs:
                metadata = doc.metadata
                similar_cases.append({
                    'case_name': metadata.get('case_name', ''),
                    'court': metadata.get('court', ''),
                    'year': metadata.get('year', ''),
                    'citation': metadata.get('citation', ''),
                    'relevance_score': 0.8,  # Placeholder - would need actual similarity scoring
                    'summary': doc.page_content[:300] + '...'
                })
            
            return {
                'total_found': len(similar_cases),
                'cases': similar_cases
            }
            
        except Exception as e:
            logger.error(f"Error getting local similar cases: {e}")
            return {'total_found': 0, 'cases': [], 'error': str(e)}
    
    def _combine_online_local_analysis(self, online_results: Dict[str, Any], 
                                     local_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine online and local case analysis"""
        
        return {
            'total_sources': 2,
            'online_available': online_results.get('status') == 'success',
            'local_cases': local_results.get('total_found', 0),
            'online_cases': len(online_results.get('search_results', [])),
            'recommendation': 'Comprehensive analysis available from both local and online sources'
        }
    
    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis"""
        
        confidence = 0.0
        
        total_cases = analysis.get('total_cases_found', 0)
        if total_cases >= 5:
            confidence += 0.4
        elif total_cases >= 2:
            confidence += 0.2
        
        verdict_patterns = analysis.get('verdict_patterns', [])
        if verdict_patterns:
            confidence += 0.3
        
        success_factors = analysis.get('success_factors', [])
        if success_factors:
            confidence += 0.2
        
        strategic_insights = analysis.get('strategic_insights', [])
        if strategic_insights:
            confidence += 0.1
        
        return min(confidence, 1.0)

async def test_enhanced_system(vectorstore, test_queries: List[str]):
    """Test the enhanced legal system with sample queries"""
    
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnableParallel, RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        
        # Initialize LLM
        llm = ChatOllama(model="llama3.2:3b", temperature=0.1)
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create enhanced prompt
        enhanced_prompt = PromptTemplate.from_template(create_enhanced_legal_prompt())
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create chain
        enhanced_chain = (
            RunnableParallel({
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            })
            | enhanced_prompt
            | llm
            | StrOutputParser()
        )
        
        # Initialize advanced analyzer
        analyzer = AdvancedCaseLawAnalyzer(vectorstore)
        
        logger.info("Testing enhanced legal system with sample queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"🔍 TEST QUERY {i}: {query}")
            print(f"{'='*80}")
            
            try:
                # Get basic enhanced response
                print("📄 **BASIC LEGAL RESPONSE:**")
                response = enhanced_chain.invoke(query)
                print(response)
                
                # Get advanced analysis
                print(f"\n⚖️ **ADVANCED CASE LAW ANALYSIS:**")
                if analyzer.llm:
                    verdict_analysis = analyzer.analyze_verdict_judgment(query)
                    print(f"Cases Analyzed: {verdict_analysis.get('total_cases_found', 0)}")
                    
                    if verdict_analysis.get('verdict_patterns'):
                        print("Verdict Patterns:")
                        for pattern in verdict_analysis['verdict_patterns']:
                            print(f"  • {pattern}")
                    
                    if verdict_analysis.get('success_factors'):
                        print("Success Factors:")
                        for factor in verdict_analysis['success_factors'][:3]:
                            print(f"  ✅ {factor}")
                    
                    if verdict_analysis.get('failure_factors'):
                        print("Common Failure Factors:")
                        for factor in verdict_analysis['failure_factors'][:3]:
                            print(f"  ❌ {factor}")
                
                # Get strategic recommendations
                print(f"\n🎯 **STRATEGIC RECOMMENDATIONS:**")
                if analyzer.llm:
                    strategy = analyzer.generate_strategy_recommendations(query)
                    if strategy.get('strategy_recommendation'):
                        print(strategy['strategy_recommendation'])
                        print(f"\nConfidence Score: {strategy.get('confidence', 0):.2f}")
                
                # Search for similar cases online
                print(f"\n� **SIMILAR CASES SEARCH:**")
                similar_cases = await analyzer.search_similar_cases_online(query, max_results=3)
                if similar_cases.get('status') == 'success':
                    print("✅ Online search successful")
                    online_results = similar_cases.get('online_results', {})
                    if online_results.get('search_results'):
                        print(f"Found {len(online_results['search_results'])} similar cases online")
                elif similar_cases.get('status') == 'unavailable':
                    print("⚠️ Online search unavailable, showing local results")
                    local_results = similar_cases.get('local_results', {})
                    print(f"Found {local_results.get('total_found', 0)} similar cases locally")
                
                # Show retrieved context
                docs = retriever.invoke(query)
                print(f"\n📚 **CASE LAW PRECEDENTS USED:**")
                for doc in docs[:3]:  # Show top 3 documents
                    case_name = doc.metadata.get('case_name', 'Unknown Case')
                    court = doc.metadata.get('court', 'Unknown Court')
                    year = doc.metadata.get('year', 'Unknown Year')
                    category = doc.metadata.get('case_category', 'Unknown Category')
                    print(f"   • {case_name} ({court}, {year}) - {category}")
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
            
            print(f"\n{'='*80}")
            await asyncio.sleep(1)  # Brief pause between queries
        
        print(f"\n🎉 **ENHANCED TESTING COMPLETE!**")
        print(f"The system now provides:")
        print(f"  ✅ Case law analysis with verdict patterns")
        print(f"  ✅ Strategic recommendations with success factors")
        print(f"  ✅ Risk assessment and failure factor analysis")
        print(f"  ✅ Online case search integration (if available)")
        print(f"  ✅ Confidence scoring for recommendations")
        
    except Exception as e:
        logger.error(f"Error testing enhanced system: {e}")

def generate_case_law_statistics(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate statistics about the case law database"""
    
    stats = {
        'total_cases': len(cases),
        'cases_by_category': {},
        'cases_by_court': {},
        'cases_by_decade': {},
        'total_keywords': set(),
        'cases_with_principles': 0,
        'cases_with_sections': 0
    }
    
    for case in cases:
        # Category statistics
        category = case.get('case_category', 'Unknown')
        stats['cases_by_category'][category] = stats['cases_by_category'].get(category, 0) + 1
        
        # Court statistics
        court = case.get('court', 'Unknown')
        stats['cases_by_court'][court] = stats['cases_by_court'].get(court, 0) + 1
        
        # Year/decade statistics
        year = case.get('year', 0)
        if year:
            decade = (year // 10) * 10
            decade_key = f"{decade}s"
            stats['cases_by_decade'][decade_key] = stats['cases_by_decade'].get(decade_key, 0) + 1
        
        # Keywords
        keywords = case.get('keywords', [])
        if isinstance(keywords, list):
            stats['total_keywords'].update(keywords)
        
        # Content analysis
        if case.get('legal_principle'):
            stats['cases_with_principles'] += 1
        
        if case.get('relevant_sections'):
            stats['cases_with_sections'] += 1
    
    stats['unique_keywords'] = len(stats['total_keywords'])
    stats['total_keywords'] = list(stats['total_keywords'])
    
    return stats

async def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Enhanced Legal Assistant Case Law Training")
    parser.add_argument("--rebuild-db", action="store_true", 
                       help="Rebuild the vector database")
    parser.add_argument("--test-queries", action="store_true",
                       help="Test the system with sample queries")
    parser.add_argument("--stats", action="store_true",
                       help="Show case law database statistics")
    
    args = parser.parse_args()
    
    # Setup environment
    project_root = setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("🏛️ AI Legal Assistant - Case Law Analysis Training")
    print("=" * 60)
    
    # Define paths
    case_law_dir = project_root / "training_data" / "case_law"
    db_path = str(project_root / "chroma_db_caselaw")
    
    # Load case law data
    logger.info("Loading case law data...")
    cases = load_case_law_data(case_law_dir)
    
    if not cases:
        logger.error("No case law data found. Please check the training_data/case_law directory.")
        sys.exit(1)
    
    # Show statistics if requested
    if args.stats:
        stats = generate_case_law_statistics(cases)
        print(f"\n📊 **Case Law Database Statistics:**")
        print(f"   Total Cases: {stats['total_cases']}")
        print(f"   Unique Keywords: {stats['unique_keywords']}")
        print(f"   Cases with Legal Principles: {stats['cases_with_principles']}")
        print(f"   Cases with Relevant Sections: {stats['cases_with_sections']}")
        
        print(f"\n📈 **Cases by Category:**")
        for category, count in sorted(stats['cases_by_category'].items()):
            print(f"   {category}: {count}")
        
        print(f"\n🏛️ **Cases by Court:**")
        for court, count in sorted(stats['cases_by_court'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {court}: {count}")
        
        return
    
    # Create documents for vector storage
    logger.info("Creating case law documents...")
    documents = create_case_law_documents(cases)
    
    if not documents:
        logger.error("No valid documents created from case law data.")
        sys.exit(1)
    
    # Setup or load vector database
    vectorstore = None
    
    if args.rebuild_db or not Path(db_path).exists():
        logger.info("Setting up vector database...")
        vectorstore = setup_vector_database(documents, db_path)
    else:
        logger.info("Loading existing vector database...")
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
            
            logger.info("Vector database loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            logger.info("Rebuilding database...")
            vectorstore = setup_vector_database(documents, db_path)
    
    if not vectorstore:
        logger.error("Failed to setup vector database.")
        sys.exit(1)
    
    # Test the system if requested
    if args.test_queries:
        test_queries = [
            "My employer is not paying overtime despite making me work 12 hours daily. What are my legal rights?",
            "I bought a defective phone online and the seller refuses to refund. What can I do legally?",
            "My husband is demanding dowry even after marriage. How can I protect myself?",
            "My child is 2 years old but birth is not registered. What is the process now?",
            "Can I file a consumer complaint for a service that was not provided properly?"
        ]
        
        await test_enhanced_system(vectorstore, test_queries)
    
    print(f"\n✅ **Enhanced Legal Assistant Setup Complete!**")
    print(f"   📚 Processed {len(cases)} case law documents")
    print(f"   🗄️ Vector database: {db_path}")
    print(f"   🎯 Ready for enhanced legal analysis with case law precedents")
    
    print(f"\n💡 **Usage Instructions:**")
    print(f"   1. Use test_enhanced_legal_assistant.py for interactive testing")
    print(f"   2. The system now provides case law analysis and strategic guidance")
    print(f"   3. Queries will include relevant precedents and actionable recommendations")
    
    print(f"\n⚖️ **Remember:** This system is for educational purposes.")
    print(f"   Always consult qualified legal professionals for official advice.")

if __name__ == "__main__":
    asyncio.run(main())
