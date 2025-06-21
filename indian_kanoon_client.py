"""
Indian Kanoon API Client for Legal Assistant - Agentic RAG Integration
=====================================================================

A production-ready client for integrating Indian Kanoon API with the AI Legal Assistant.
Features budget tracking, error handling, caching, and agentic RAG flow integration.

Key Features:
- Query classification and routing
- Context-aware case law search
- Budget management with cost optimization
- Response synthesis for legal guidance
- Urgency-based prioritization

Author: AI Legal Assistant Team
Date: June 2025
"""

import os
import json
import logging
import time
import asyncio
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

# Import the official Indian Kanoon API client
try:
    from indian_kanoon.ikapi import IKApi, FileStorage, setup_logging
    IKAPI_AVAILABLE = True
except ImportError:
    try:
        from ikapi import IKApi, FileStorage, setup_logging
        IKAPI_AVAILABLE = True
    except ImportError:
        IKAPI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums for Query Classification (following mermaid flow)
class QueryType(Enum):
    CONSUMER_PROTECTION = "consumer_protection"
    CRIMINAL_LAW = "criminal_law"
    FAMILY_LAW = "family_law"
    PROPERTY_LAW = "property_law"
    CIVIL_LAW = "civil_law"
    URGENT_THREAT = "urgent_threat"
    GENERAL = "general"

class UrgencyLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class SearchContext(Enum):
    GENERAL = "general"
    CASE_LAW = "case_law"
    PROCEDURAL = "procedural"
    EMERGENCY = "emergency"

@dataclass
class QueryClassification:
    """Enhanced query classification for agentic flow"""
    query_type: QueryType
    urgency: UrgencyLevel
    confidence: float
    requires_legal_counsel: bool
    jurisdiction: str = "india"
    keywords: List[str] = field(default_factory=list)
    search_context: SearchContext = SearchContext.GENERAL
    priority_score: int = 0
    legal_sections: List[str] = field(default_factory=list)

@dataclass
class EnhancedSearchResult:
    """Enhanced search result with agentic context"""
    doc_id: str
    title: str
    court: str
    date: str
    snippet: str = ""
    position: int = 0
    relevance_score: float = 0.0
    query_type_match: bool = False
    urgency_indicators: List[str] = field(default_factory=list)
    legal_concepts: List[str] = field(default_factory=list)
    
@dataclass 
class ContextualCaseDocument:
    """Case document with enhanced context for RAG"""
    doc_id: str
    title: str
    content: str
    court: str
    date: str
    citations: List[str] = field(default_factory=list)
    cited_by: List[str] = field(default_factory=list)
    legal_principles: List[str] = field(default_factory=list)
    applicable_sections: List[str] = field(default_factory=list)
    case_type: str = "general"
    outcome: str = ""

@dataclass
class AgenticSearchStrategy:
    """Search strategy based on query classification"""
    primary_keywords: List[str]
    secondary_keywords: List[str]
    legal_sections: List[str]
    case_type_filter: str
    max_results: int
    search_depth: str  # "shallow", "deep", "comprehensive"
    priority_courts: List[str] = field(default_factory=list)

@dataclass
class SearchResult:
    """Represents a case law search result"""
    doc_id: str
    title: str
    court: str
    date: str
    snippet: str = ""
    position: int = 0
    
@dataclass 
class CaseDocument:
    """Represents a full case law document"""
    doc_id: str
    title: str
    content: str
    court: str
    date: str
    citations: List[str]
    cited_by: List[str]

class BudgetTracker:
    """Track API usage and budget"""
    
    def __init__(self, budget_limit: float = 500.0):
        self.budget_limit = budget_limit
        self.current_spending = 0.0
        self.search_count = 0
        self.document_count = 0
        self.meta_count = 0
        
        # Cost per operation (in Rs)
        self.cost_per_search = 0.5  
        self.cost_per_document = 0.25
        self.cost_per_meta = 0.10
        
    def can_afford(self, operation: str, count: int = 1) -> bool:
        """Check if we can afford an operation"""
        costs = {
            'search': self.cost_per_search * count,
            'document': self.cost_per_document * count,
            'meta': self.cost_per_meta * count
        }
        
        cost = costs.get(operation, 0)
        return (self.current_spending + cost) <= self.budget_limit
    
    def record_usage(self, operation: str, count: int = 1):
        """Record API usage"""
        costs = {
            'search': self.cost_per_search * count,
            'document': self.cost_per_document * count,
            'meta': self.cost_per_meta * count
        }
        
        cost = costs.get(operation, 0)
        self.current_spending += cost
        
        if operation == 'search':
            self.search_count += count
        elif operation == 'document':
            self.document_count += count
        elif operation == 'meta':
            self.meta_count += count
            
        logger.info(f"API Usage - {operation}: +{cost:.2f} Rs, Total: {self.current_spending:.2f} Rs")
    
    def get_status(self) -> Dict[str, Any]:
        """Get budget status"""
        return {
            'budget_limit': self.budget_limit,
            'current_budget': self.current_spending,
            'remaining_budget': self.budget_limit - self.current_spending,
            'budget_percentage_used': (self.current_spending / self.budget_limit) * 100,
            'search_count': self.search_count,
            'document_count': self.document_count,
            'meta_count': self.meta_count
        }

class IndianKanoonClient:
    """
    Production-ready Indian Kanoon API client with agentic RAG integration.
    Implements intelligent query routing, context-aware search, and response synthesis.
    """
    
    def __init__(self, api_token: str, budget_limit: float = 500.0, data_dir: str = "temp_data"):
        """
        Initialize the Indian Kanoon client with agentic capabilities.
        
        Args:
            api_token: Indian Kanoon API token
            budget_limit: Budget limit in Rs (default: 500)
            data_dir: Directory for temporary data storage
        """
        if not IKAPI_AVAILABLE:
            raise ImportError("Indian Kanoon API client (ikapi) not available. Please check installation.")
        
        self.api_token = api_token
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
          # Initialize budget tracker
        self.budget = BudgetTracker(budget_limit)        # Initialize agentic components
        self.query_classifier = self.QueryClassifier()
        self.search_engine = self.AgenticSearchEngine()
        
        # Initialize the official API client
        self._init_api_client()
        
        # Enhanced caching with classification-based keys
        self.search_cache = {}
        self.doc_cache = {}
        self.classification_cache = {}
        
        logger.info(f"Agentic Indian Kanoon client initialized with budget limit: Rs {budget_limit}")
    
    def _init_api_client(self):
        """Initialize the official Indian Kanoon API client"""
        try:
            # Create a simple args object for the API client
            class Args:
                def __init__(self, token, datadir):
                    self.token = token
                    self.datadir = str(datadir)
                    self.maxcites = 5  # Limit citations to control costs
                    self.maxcitedby = 5  # Limit cited by to control costs
                    self.orig = False  # Don't download original documents
                    self.maxpages = 1  # Limit to 1 page for cost control
                    self.pathbysrc = False
                    self.numworkers = 1
                    self.addedtoday = False
                    self.fromdate = None
                    self.todate = None
                    self.sortby = None
            
            args = Args(self.api_token, self.data_dir)
            storage = FileStorage(str(self.data_dir))
            
            self.api_client = IKApi(args, storage)            
            logger.info("Official Indian Kanoon API client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Indian Kanoon API client: {e}")
            raise
    
    async def agentic_search(self, query: str, user_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform agentic search following the mermaid flow diagram.
        
        Args:
            query: User's legal query
            user_context: Additional context about user's situation
            
        Returns:
            Dictionary containing classification, search results, and recommendations
        """
        try:
            # Step 1: Query Classification (following mermaid flow)
            classification = self.classify_query(query)
            logger.info(f"Query classified as {classification.query_type.value} with urgency {classification.urgency.value}")
            
            # Step 2: Urgency Assessment & Routing
            if classification.urgency == UrgencyLevel.CRITICAL:
                return await self._handle_critical_query(query, classification, user_context)
            
            # Step 3: Knowledge Retrieval Strategy
            search_strategy = self.search_engine.create_search_strategy(classification, user_context)
            
            # Step 4: Execute Search
            search_results = await self._execute_strategic_search(query, search_strategy, classification)
            
            # Step 5: Response Synthesis
            response = self._synthesize_agentic_response(query, classification, search_results, user_context)
            
            return response
            
        except Exception as e:
            logger.error(f"Agentic search failed: {e}")
            return self.generate_error_response(query, str(e))
    
    async def _execute_strategic_search(self, query: str, strategy: AgenticSearchStrategy, 
                                       classification: QueryClassification) -> List[Dict[str, Any]]:
        """Execute search based on strategy"""
        try:
            # Create enhanced query with strategy keywords
            enhanced_query = self._enhance_query_with_strategy(query, strategy)
            
            # Perform API search
            search_results = await self._perform_api_search(enhanced_query, strategy.max_results)
            
            # Filter and rank results based on strategy
            filtered_results = self._filter_and_rank_results(search_results, classification, strategy)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Strategic search failed: {e}")
            # Return empty results on failure
            return []
    
    def _enhance_query_with_strategy(self, query: str, strategy: AgenticSearchStrategy) -> str:
        """Enhance query with strategic keywords"""
        enhanced_parts = [query]
        
        # Add primary keywords
        if strategy.primary_keywords:
            enhanced_parts.extend(strategy.primary_keywords[:2])  # Add top 2 primary keywords
        
        # Add legal sections for better context
        if strategy.legal_sections:
            enhanced_parts.append(strategy.legal_sections[0])  # Add most relevant legal section
        
        return " ".join(enhanced_parts)
    
    def _filter_and_rank_results(self, results: List[Dict[str, Any]], 
                                classification: QueryClassification, 
                                strategy: AgenticSearchStrategy) -> List[Dict[str, Any]]:
        """Filter and rank results based on classification and strategy"""
        if not results:
            return []
        
        # Add relevance scores
        for result in results:
            result["relevance_score"] = self._calculate_relevance(result, classification)
        
        # Sort by relevance score
        sorted_results = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Apply strategy filters
        filtered_results = []
        for result in sorted_results:
            # Court priority filtering
            if strategy.priority_courts:
                court = result.get('court', '').lower()
                if any(priority_court.lower() in court for priority_court in strategy.priority_courts):
                    result["relevance_score"] += 0.5  # Boost for priority courts
            
            filtered_results.append(result)
        
        # Re-sort after filtering
        filtered_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return filtered_results[:strategy.max_results]
    
    async def _perform_api_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform the actual API search"""
        try:
            # Check budget
            if not self.budget.can_afford('search', 1):
                logger.warning(f"Budget exceeded - cannot perform search")
                return []
            
            # Check cache first
            cache_key = f"search_{hash(query)}"
            if cache_key in self.search_cache:
                logger.info(f"Returning cached results for query: {query[:50]}...")
                return self.search_cache[cache_key]
            
            # Perform search using the official API
            logger.info(f"Performing API search for: {query[:50]}...")
            
            # Create search parameters
            search_params = {
                'query': query,
                'num': min(max_results, 20),  # Limit to reasonable number
                'sort': 'relevance'
            }
            
            # Use the official API client to search
            if hasattr(self.api_client, 'search'):
                api_results = self.api_client.search(**search_params)
            else:
                # Fallback search method
                api_results = await self._fallback_search(query, max_results)
            
            # Record usage
            self.budget.record_usage('search', 1)
            
            # Convert API results to our format
            results = self._convert_api_results(api_results)
            
            # Cache results
            self.search_cache[cache_key] = results
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"API search failed: {e}")
            return []
    
    def _convert_api_results(self, api_results: Any) -> List[Dict[str, Any]]:
        """Convert API results to our standard format"""
        results = []
        
        try:
            # Handle different API result formats
            if hasattr(api_results, '__iter__') and not isinstance(api_results, (str, bytes)):
                # Iterable results
                for i, result in enumerate(api_results):
                    if isinstance(result, dict):
                        # Already in dict format
                        converted = self._normalize_result_dict(result, i)
                        results.append(converted)
                    else:
                        # Convert object to dict
                        converted = self._convert_result_object(result, i)
                        results.append(converted)
            else:
                # Single result or unexpected format
                logger.warning(f"Unexpected API result format: {type(api_results)}")
                
        except Exception as e:
            logger.error(f"Failed to convert API results: {e}")
        
        return results
    
    def _normalize_result_dict(self, result: Dict[str, Any], position: int) -> Dict[str, Any]:
        """Normalize a result dictionary to our standard format"""
        return {
            "doc_id": str(result.get('doc_id', result.get('id', f'doc_{position}'))),
            "title": str(result.get('title', result.get('name', 'Untitled'))),
            "court": str(result.get('court', result.get('jurisdiction', 'Unknown Court'))),
            "date": str(result.get('date', result.get('judgement_date', 'Unknown Date'))),
            "snippet": str(result.get('snippet', result.get('summary', str(result.get('content', ''))[:200] + '...'))),
            "position": position,
            "relevance_score": 0.0,
            "url": result.get('url', ''),
            "citations": result.get('citations', []),
            "cited_by": result.get('cited_by', [])
        }
    
    def _convert_result_object(self, result: Any, position: int) -> Dict[str, Any]:
        """Convert a result object to our standard format"""
        try:
            return {
                "doc_id": str(getattr(result, 'doc_id', getattr(result, 'id', f'doc_{position}'))),
                "title": str(getattr(result, 'title', getattr(result, 'name', 'Untitled'))),
                "court": str(getattr(result, 'court', getattr(result, 'jurisdiction', 'Unknown Court'))),
                "date": str(getattr(result, 'date', getattr(result, 'judgement_date', 'Unknown Date'))),
                "snippet": str(getattr(result, 'snippet', getattr(result, 'summary', str(result)[:200] + '...'))),
                "position": position,
                "relevance_score": 0.0,
                "url": getattr(result, 'url', ''),
                "citations": getattr(result, 'citations', []),
                "cited_by": getattr(result, 'cited_by', [])
            }
        except Exception as e:
            logger.error(f"Failed to convert result object: {e}")
            return {
                "doc_id": f"doc_{position}",
                "title": "Conversion Error",
                "court": "Unknown Court",
                "date": "Unknown Date",
                "snippet": str(result)[:200] if result else "No content",
                "position": position,
                "relevance_score": 0.0,
                "url": "",
                "citations": [],
                "cited_by": []
            }
    
    async def _fallback_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Fallback search when official API is not available"""
        logger.warning("Using fallback search - limited functionality")
        
        # Create mock results for demonstration
        fallback_results = []
        for i in range(min(max_results, 3)):  # Limit fallback results
            fallback_results.append({
                "doc_id": f"fallback_doc_{i}",
                "title": f"Fallback Legal Case {i+1} - {query[:30]}",
                "court": "Fallback Court",
                "date": "2024-01-01",
                "snippet": f"This is a fallback result for query: {query}. Please check your API configuration.",
                "position": i,
                "url": "",
                "citations": [],
                "cited_by": []
            })
        
        return fallback_results
    
    def clear_cache(self):
        """Clear all cached data"""
        self.search_cache.clear()
        self.doc_cache.clear()
        self.classification_cache.clear()
        logger.info("Caches cleared")
    
    def classify_query(self, query: str) -> QueryClassification:
        """Classify query with caching"""
        cache_key = f"classify_{hash(query)}"
        
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]
        
        classification = self.query_classifier.classify_query(query)
        self.classification_cache[cache_key] = classification
        
        return classification
    
    async def _handle_critical_query(self, query: str, classification: QueryClassification, 
                                   user_context: Optional[str]) -> Dict[str, Any]:
        """Handle critical/urgent queries with immediate response protocol"""
        logger.warning(f"CRITICAL QUERY DETECTED: {query}")
        
        # Emergency search with expanded results
        emergency_results = await self._emergency_search(query, classification)
        
        return {
            "classification": {
                "query_type": classification.query_type.value,
                "urgency": "CRITICAL",
                "requires_immediate_counsel": True,
                "priority_score": classification.priority_score
            },
            "emergency_response": {
                "immediate_actions": self._get_immediate_actions(classification.query_type),
                "emergency_contacts": self._get_emergency_contacts(classification.query_type),
                "legal_protections": self._get_legal_protections(classification.query_type)
            },
            "search_results": emergency_results,
            "advocate_recommendation": True,
            "disclaimer": "⚠️ URGENT LEGAL MATTER - Seek immediate legal counsel. This is informational only."
        }
    
    async def _emergency_search(self, query: str, classification: QueryClassification) -> List[Dict[str, Any]]:
        """Perform emergency search with priority results"""
        emergency_keywords = [
            "immediate relief", "urgent", "anticipatory bail", "protection order",
            "emergency provisions", "interim relief"
        ]
        
        enhanced_query = f"{query} {' '.join(emergency_keywords[:3])}"
        
        # Increase search depth for critical queries
        results = await self._perform_api_search(enhanced_query, max_results=20)
        
        # Filter and prioritize results
        return self._prioritize_emergency_results(results, classification)
    
    def _prioritize_emergency_results(self, results: List[Dict[str, Any]], classification: QueryClassification) -> List[Dict[str, Any]]:
        """Prioritize results for emergency queries"""
        # Tag results with emergency relevance scores
        for result in results:
            result["emergency_score"] = self._calculate_emergency_relevance(result, classification)
            result["relevance_score"] = result["emergency_score"]  # Use emergency score as main relevance
            
        # Sort by emergency score
        sorted_results = sorted(results, key=lambda x: x.get("emergency_score", 0), reverse=True)
        
        # Return top results (higher limit for emergency situations)
        return sorted_results[:10]
    
    def _calculate_emergency_relevance(self, result: Dict[str, Any], classification: QueryClassification) -> float:
        """Calculate emergency relevance score"""
        score = self._calculate_relevance(result, classification)  # Base relevance
        
        # Additional emergency-specific scoring
        content = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
        
        # Emergency keywords with weights
        emergency_indicators = {
            "immediate": 2.0,
            "urgent": 2.0,
            "emergency": 2.0,
            "protection": 1.5,
            "relief": 1.5,
            "interim": 1.5,
            "bail": 1.5,
            "safety": 1.5,
            "danger": 2.0,
            "threat": 2.0,
        }
        
        # Score based on emergency keywords
        for keyword, weight in emergency_indicators.items():
            if keyword in content:
                score += weight
        
        return score
    
    def _get_immediate_actions(self, query_type: QueryType) -> List[str]:
        """Get immediate actions based on query type"""
        common_actions = [
            "Contact a local lawyer immediately",
            "Document all relevant evidence and communications"
        ]
        
        actions_by_type = {
            QueryType.CRIMINAL_LAW: [
                "If facing arrest, apply for anticipatory bail",
                "Consult with a criminal defense lawyer",
                "Do not make statements to police without a lawyer present"
            ],
            QueryType.FAMILY_LAW: [
                "Consider obtaining a protection order if facing threats",
                "Secure important documents and financial information",
                "Ensure children's safety and document any incidents"
            ],
            QueryType.CONSUMER_PROTECTION: [
                "File a formal complaint with the service provider/retailer",
                "Contact consumer protection authorities",
                "Document all defects and communication with seller"
            ],
            QueryType.PROPERTY_LAW: [
                "File a police complaint if there's forcible dispossession",
                "Secure property documents and proof of ownership",
                "Consider filing for emergency injunction"
            ]
        }
        
        specific_actions = actions_by_type.get(query_type, [])
        return common_actions + specific_actions[:3]  # Combine common + type-specific actions
    
    def _get_emergency_contacts(self, query_type: QueryType) -> List[Dict[str, str]]:
        """Get relevant emergency contacts based on query type"""
        common_contacts = [
            {"name": "Police Emergency", "number": "100"},
            {"name": "Legal Services Authority", "number": "1516"}
        ]
        
        contacts_by_type = {
            QueryType.CRIMINAL_LAW: [
                {"name": "National Legal Services Authority", "number": "1516"},
                {"name": "Women's Helpline", "number": "181"}
            ],
            QueryType.FAMILY_LAW: [
                {"name": "Women's Helpline", "number": "181"},
                {"name": "Child Helpline", "number": "1098"}
            ],
            QueryType.CONSUMER_PROTECTION: [
                {"name": "Consumer Helpline", "number": "1800-11-4000"},
                {"name": "State Consumer Dispute Redressal Commission", "number": "Varies by state"}
            ],
            QueryType.PROPERTY_LAW: [
                {"name": "Local Police Station", "number": "Check local directory"},
                {"name": "Municipal Corporation", "number": "Varies by city"}
            ]
        }
        
        specific_contacts = contacts_by_type.get(query_type, [])
        return common_contacts + specific_contacts
    
    def _get_legal_protections(self, query_type: QueryType) -> List[str]:
        """Get relevant legal protections based on query type"""
        common_protections = [
            "Constitutional rights under Article 21 (Right to life and liberty)",
            "Right to legal representation"
        ]
        
        protections_by_type = {
            QueryType.CRIMINAL_LAW: [
                "Right to silence under Article 20(3)",
                "Protection against arbitrary arrest under Article 22",
                "Right to fair trial under Article 21"
            ],
            QueryType.FAMILY_LAW: [
                "Protection of Women from Domestic Violence Act, 2005",
                "Maintenance under Section 125 of CrPC",
                "Child custody protections under Guardian and Wards Act"
            ],
            QueryType.CONSUMER_PROTECTION: [
                "Consumer Protection Act, 2019 protections",
                "Right to compensation for defective products/services",
                "Right to file class action suits"
            ],
            QueryType.PROPERTY_LAW: [
                "Protection against forcible dispossession",
                "Right to peaceful possession under Section 6 of Specific Relief Act",
                "Right to injunction against interference"
            ]
        }
        
        specific_protections = protections_by_type.get(query_type, [])
        return common_protections + specific_protections
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate search results based on document ID"""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            doc_id = result.get("doc_id")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(result)
        
        return unique_results
    
    def _synthesize_agentic_response(self, query: str, classification: QueryClassification, 
                                   search_results: List[Dict[str, Any]], user_context: Optional[str]) -> Dict[str, Any]:
        """Synthesize final response with agentic insights"""
        # Base response structure
        response = {
            "query": query,
            "classification": {
                "query_type": classification.query_type.value,
                "urgency": classification.urgency.value,
                "requires_legal_counsel": classification.requires_legal_counsel
            },
            "search_results": search_results,
            "recommendations": self._generate_recommendations(classification, search_results),
            "disclaimers": [
                "This information is for educational purposes only and does not constitute legal advice.",
                "Consult with a qualified lawyer for specific legal guidance."
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        # Add user context analysis if available
        if user_context:
            response["context_analysis"] = {
                "relevant_factors": self._extract_context_factors(user_context),
                "potential_issues": self._identify_potential_issues(user_context, classification)
            }
        
        # Add next steps based on classification
        response["next_steps"] = self._generate_next_steps(classification, search_results)
        
        return response
    
    def _generate_recommendations(self, classification: QueryClassification, 
                               search_results: List[Dict[str, Any]]) -> List[str]:
        """Generate tailored recommendations based on query classification and results"""
        recommendations = []
        
        # Add general recommendations based on query type
        query_type_recs = {
            QueryType.CONSUMER_PROTECTION: [
                "File a formal complaint with the business first",
                "Approach the consumer forum if the complaint is not resolved",
                "Gather all bills, warranties, and communication as evidence"
            ],
            QueryType.CRIMINAL_LAW: [
                "Consult with a criminal lawyer immediately",
                "Do not give statements without legal representation",
                "Understand your rights during investigation and trial"
            ],
            QueryType.FAMILY_LAW: [
                "Consider mediation for family disputes",
                "Consult a family law specialist",
                "Focus on documentation and maintaining records"
            ],
            QueryType.PROPERTY_LAW: [
                "Verify property documents and ownership history",
                "Consider getting a legal title search done",
                "File appropriate documentation with relevant authorities"
            ]
        }
        
        # Add recommendations based on query type
        type_specific = query_type_recs.get(classification.query_type, [])
        recommendations.extend(type_specific)
        
        # Add case-law based recommendations if available
        if search_results:
            recommendations.append(f"Review similar cases from {search_results[0].get('court', 'courts')}")
        
        # Add urgency-based recommendations
        if classification.urgency in [UrgencyLevel.HIGH, UrgencyLevel.MEDIUM]:
            recommendations.insert(0, "Seek legal counsel promptly due to time-sensitive nature")
        
        return recommendations
    
    def _extract_context_factors(self, user_context: str) -> List[str]:
        """Extract relevant factors from user context"""
        # This is a simplified implementation
        # In production, this would use NLP to extract entities and context
        factors = []
        
        # Look for common legal factors in context
        if "contract" in user_context.lower():
            factors.append("Contractual relationship")
        if "injury" in user_context.lower() or "damage" in user_context.lower():
            factors.append("Potential damages or injuries")
        if "evidence" in user_context.lower() or "proof" in user_context.lower():
            factors.append("Evidentiary considerations")
        if "time" in user_context.lower() or "deadline" in user_context.lower():
            factors.append("Time-sensitive elements")
        
        return factors
    
    def _identify_potential_issues(self, user_context: str, classification: QueryClassification) -> List[str]:
        """Identify potential legal issues from user context"""
        # Simplified implementation
        issues = []
        
        # Check for common legal issues
        if "deadline" in user_context.lower() or "limitation" in user_context.lower():
            issues.append("Potential statute of limitations concerns")
        if "jurisdiction" in user_context.lower() or "state" in user_context.lower():
            issues.append("Jurisdictional considerations")
        if "document" in user_context.lower():
            issues.append("Documentation requirements")
        
        return issues
    
    def _generate_next_steps(self, classification: QueryClassification, 
                          search_results: List[Dict[str, Any]]) -> List[str]:
        """Generate concrete next steps based on classification"""
        next_steps = []
        
        # Basic steps based on urgency
        if classification.urgency == UrgencyLevel.HIGH:
            next_steps.append("📋 Consult with a lawyer within 24-48 hours")
        elif classification.urgency == UrgencyLevel.MEDIUM:
            next_steps.append("📋 Schedule a legal consultation within the next week")
        else:
            next_steps.append("📋 Research your legal options and document your situation")
        
        # Add query type specific steps
        query_type_steps = {
            QueryType.CONSUMER_PROTECTION: [
                "📋 Gather all receipts, warranty information, and product documentation",
                "📋 Write a formal complaint letter to the business",
                "📋 Research the consumer dispute resolution process in your jurisdiction"
            ],
            QueryType.CRIMINAL_LAW: [
                "📋 Do not discuss your case with anyone except your lawyer",
                "📋 Collect all relevant documents and evidence",
                "📋 Understand the charges and potential consequences"
            ],
            QueryType.FAMILY_LAW: [
                "📋 Gather all financial documents and records",
                "📋 Document all communications related to the dispute",
                "📋 Consider the welfare of any children involved"
            ],
            QueryType.PROPERTY_LAW: [
                "📋 Verify property records at the local registry",
                "📋 Check for any encumbrances or liens on the property",
                "📋 Ensure all property taxes and dues are up to date"
            ]
        }
        
        # Add type-specific steps
        type_steps = query_type_steps.get(classification.query_type, [])
        next_steps.extend(type_steps)
        
        # Add result-based steps if available
        if search_results:
            court = search_results[0].get('court', 'the relevant court')
            next_steps.append(f"📋 Research similar cases from {court} to understand precedent")
        
        return next_steps
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.clear_cache()
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        self.clear_cache()
        
    def _calculate_relevance(self, result: Dict[str, Any], classification: QueryClassification) -> float:
        """Calculate relevance score for search result"""
        score = 0.0
        
        # Keyword matching in title and snippet
        content = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
        
        for keyword in classification.keywords:
            if keyword.lower() in content:
                score += 1.0
        
        # Court priority scoring
        court = result.get('court', '').lower()
        if 'supreme court' in court:
            score += 2.0
        elif 'high court' in court:
            score += 1.5
        elif 'district' in court:
            score += 1.0
            
        # Date recency scoring
        date = result.get('date', '')
        try:
            if date and len(date) > 4:  # Valid date format
                year = int(date[-4:])  # Extract year from end of date string
                current_year = datetime.now().year
                # More recent cases get higher scores
                recency_score = 1.0 - (0.05 * (current_year - year))  # 5% deduction per year
                score += max(0, recency_score)  # Ensure non-negative
        except (ValueError, TypeError):
            pass  # Ignore date parsing errors
            
        return score
    
    def generate_error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Generate error response when search fails"""
        return {
            "error": True,
            "message": f"Search failed: {error}",
            "query": query,
            "recommendations": [
                "Try rephrasing your query",
                "Check your internet connection",
                "Contact support if the issue persists"
            ],
            "fallback_guidance": [
                "Consult with a local lawyer",
                "Visit your nearest legal aid center",
                "Check legal aid websites for similar cases"
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    # Legacy method for backward compatibility
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Legacy search method for backward compatibility.
        For new implementations, use agentic_search() instead.
        """
        try:
            # Convert to async call
            import asyncio
            if asyncio.get_event_loop().is_running():
                # If already in async context, create new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(self._perform_api_search(query, max_results))
                loop.close()
            else:
                results = asyncio.run(self._perform_api_search(query, max_results))
            
            # Convert to SearchResult objects for backward compatibility
            search_results = []
            for result in results:
                search_result = SearchResult(
                    doc_id=result["doc_id"],
                    title=result["title"],
                    court=result["court"],
                    date=result["date"],
                    snippet=result["snippet"],
                    position=result["position"]
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
            
    # Nested utility classes for internal use
    class QueryClassifier:
        """Intelligent query classification for agentic routing"""
        
        def __init__(self):
            # Legal keywords mapping for classification
            self.classification_rules = {
                QueryType.CONSUMER_PROTECTION: {
                    "keywords": ["purchase", "bottle", "price", "consumer", "overcharge", "defective product", 
                               "warranty", "refund", "service deficiency", "product liability"],
                    "legal_sections": ["Consumer Protection Act 2019", "Legal Metrology Act"],
                    "urgency_indicators": ["financial loss", "health risk"]
                },
                QueryType.CRIMINAL_LAW: {
                    "keywords": ["rape case", "false case", "extortion", "blackmail", "threat", "assault", 
                               "harassment", "dowry", "domestic violence", "fraud"],
                    "legal_sections": ["IPC", "CrPC", "Protection of Women Act"],
                    "urgency_indicators": ["immediate threat", "safety risk", "ongoing harassment"]
                },
                QueryType.FAMILY_LAW: {
                    "keywords": ["divorce", "marriage", "custody", "maintenance", "alimony", "adoption",
                               "property settlement", "domestic disputes"],
                    "legal_sections": ["Hindu Marriage Act", "Special Marriage Act", "Family Courts Act"],
                    "urgency_indicators": ["child welfare", "domestic violence"]
                },
                QueryType.PROPERTY_LAW: {
                    "keywords": ["property", "land", "ownership", "title", "registration", "partition",
                               "easement", "encroachment", "possession"],
                    "legal_sections": ["Transfer of Property Act", "Registration Act", "Land Acquisition Act"],
                    "urgency_indicators": ["illegal occupation", "forced eviction"]
                }
            }
        
        def classify_query(self, query: str) -> QueryClassification:
            """Classify query using rule-based approach with scoring"""
            query_lower = query.lower()
            scores = {}
            
            # Calculate scores for each query type
            for query_type, rules in self.classification_rules.items():
                score = 0
                matched_keywords = []
                urgency_indicators = []
                
                # Keyword matching
                for keyword in rules["keywords"]:
                    if keyword in query_lower:
                        score += 1
                        matched_keywords.append(keyword)
                
                # Urgency detection
                for indicator in rules["urgency_indicators"]:
                    if indicator in query_lower:
                        urgency_indicators.append(indicator)
                        score += 2  # Higher weight for urgency
                
                if score > 0:
                    scores[query_type] = {
                        "score": score,
                        "keywords": matched_keywords,
                        "urgency_indicators": urgency_indicators
                    }
            
            # Determine best match
            if not scores:
                return self._default_classification(query)
            
            best_match = max(scores.items(), key=lambda x: x[1]["score"])
            query_type, match_data = best_match
            
            # Determine urgency level
            urgency = self._calculate_urgency(query_lower, match_data["urgency_indicators"])
            
            # Determine if legal counsel is required
            requires_legal_counsel = self._needs_legal_counsel(query_lower, urgency)
            
            # Calculate confidence based on score
            confidence = min(0.5 + (best_match[1]["score"] * 0.1), 0.95)
            
            # Calculate priority score
            priority_score = best_match[1]["score"]
            if urgency == UrgencyLevel.HIGH:
                priority_score += 10
            elif urgency == UrgencyLevel.MEDIUM:
                priority_score += 5
            
            return QueryClassification(
                query_type=query_type,
                keywords=match_data["keywords"],
                urgency=urgency,
                requires_legal_counsel=requires_legal_counsel,
                confidence=confidence,
                jurisdiction="india",
                search_context=SearchContext.GENERAL,
                priority_score=priority_score,
                legal_sections=self.classification_rules[query_type]["legal_sections"]
            )
        
        def _default_classification(self, query: str) -> QueryClassification:
            """Provide default classification when no rules match"""
            # Default to general legal query with low urgency
            return QueryClassification(
                query_type=QueryType.GENERAL,
                keywords=[],
                urgency=UrgencyLevel.LOW,
                requires_legal_counsel=False,
                confidence=0.5,
                jurisdiction="india",
                search_context=SearchContext.GENERAL,
                priority_score=0,
                legal_sections=[]
            )
        
        def _calculate_urgency(self, query: str, urgency_indicators: List[str]) -> UrgencyLevel:
            """Determine urgency level based on indicators and language"""
            # Check for explicit urgency markers
            explicit_urgency = ["urgent", "emergency", "immediate", "critical", "life threatening"]
            
            if any(marker in query for marker in explicit_urgency) or len(urgency_indicators) > 1:
                return UrgencyLevel.HIGH
            elif urgency_indicators:
                return UrgencyLevel.MEDIUM
            else:
                return UrgencyLevel.LOW
        
        def _needs_legal_counsel(self, query: str, urgency: UrgencyLevel) -> bool:
            """Determine if query needs professional legal counsel"""
            legal_counsel_indicators = [
                "court case", "lawsuit", "legal notice", "sue", "trial", "hearing date", 
                "summons", "warrant", "police complaint", "fir", "arrest", "bail"
            ]
            
            # Complex legal issues or high urgency situations likely need a lawyer
            return any(indicator in query for indicator in legal_counsel_indicators) or urgency == UrgencyLevel.HIGH
    
    class AgenticSearchEngine:
        """Search engine that creates strategies based on query classification"""
        
        def create_search_strategy(self, classification: QueryClassification, 
                             user_context: Optional[str] = None) -> AgenticSearchStrategy:
            """Create search strategy based on classification"""
            
            # Base strategy
            strategy = AgenticSearchStrategy(
                primary_keywords=classification.keywords[:3],  # Top 3 keywords
                secondary_keywords=[],
                legal_sections=[],
                case_type_filter="all",
                max_results=5,
                search_depth="moderate",
                priority_courts=[]
            )
            
            # Customize based on query type
            if classification.query_type == QueryType.CONSUMER_PROTECTION:
                strategy.legal_sections = ["Consumer Protection Act", "Sale of Goods Act"]
                strategy.secondary_keywords = ["consumer rights", "deficiency", "unfair trade"]
                strategy.priority_courts = ["National Consumer Disputes Redressal Commission", "Supreme Court"]
                
            elif classification.query_type == QueryType.CRIMINAL_LAW:
                strategy.legal_sections = ["Indian Penal Code", "Code of Criminal Procedure"]
                strategy.secondary_keywords = ["bail", "sections", "criminal law"]
                strategy.priority_courts = ["Supreme Court", "High Court"]
                strategy.search_depth = "deep"  # More thorough for criminal matters
                
            elif classification.query_type == QueryType.FAMILY_LAW:
                strategy.legal_sections = ["Hindu Marriage Act", "Special Marriage Act", "Guardians and Wards Act"]
                strategy.secondary_keywords = ["family court", "marriage", "maintenance"]
                strategy.priority_courts = ["Family Court", "District Court"]
                
            elif classification.query_type == QueryType.PROPERTY_LAW:
                strategy.legal_sections = ["Transfer of Property Act", "Registration Act"]
                strategy.secondary_keywords = ["property rights", "possession", "title"]
                strategy.priority_courts = ["Civil Court", "District Court"]
            
            # Adjust based on urgency
            if classification.urgency == UrgencyLevel.CRITICAL:
                strategy.max_results = 10  # More results for urgent cases
                strategy.search_depth = "comprehensive"
                strategy.secondary_keywords.extend(["urgent", "immediate", "emergency"])
                
            elif classification.urgency == UrgencyLevel.HIGH:
                strategy.max_results = 8
                strategy.search_depth = "deep"
            
            return strategy

# Factory function for creating Indian Kanoon client
def create_indian_kanoon_client(api_token: str, budget_limit: float = 500.0, 
                               data_dir: str = "temp_data") -> IndianKanoonClient:
    """
    Factory function to create an Indian Kanoon client.
    
    Args:
        api_token: Indian Kanoon API token
        budget_limit: Budget limit in Rs (default: 500)
        data_dir: Directory for temporary data storage
        
    Returns:
        IndianKanoonClient instance
    """
    return IndianKanoonClient(api_token, budget_limit, data_dir)


# Utility function for agentic legal search
async def agentic_legal_search(api_token: str, query: str, 
                              user_context: Optional[str] = None,
                              budget_limit: float = 500.0) -> Dict[str, Any]:
    """
    Perform agentic legal search using the Indian Kanoon API.
    
    Args:
        api_token: Indian Kanoon API token
        query: User's legal query
        user_context: Additional context about user's situation
        budget_limit: Budget limit in Rs
        
    Returns:
        Dictionary containing search results and recommendations
    """
    try:
        client = create_indian_kanoon_client(api_token, budget_limit)
        return await client.agentic_search(query, user_context)
    except Exception as e:
        logger.error(f"Agentic legal search failed: {e}")
        return {
            'error': str(e),
            'query': query,
            'classification': {'query_type': 'unknown', 'urgency': 'unknown'},
            'search_results': {'results': []},
            'recommendations': []
        }


# Query classifier for standalone use
class QueryClassifier:
    """Standalone query classifier following the mermaid flow"""
    
    def classify_query(self, query: str) -> QueryClassification:
        """Classify a legal query into type and urgency level"""
        query_lower = query.lower()
        
        # Initialize classification
        query_type = QueryType.GENERAL
        urgency = UrgencyLevel.LOW
        confidence = 0.0
        requires_legal_counsel = False
        keywords = []
        
        # Consumer Protection Keywords
        consumer_keywords = [
            'defective', 'product', 'refund', 'warranty', 'consumer', 
            'service', 'complaint', 'shop', 'buy', 'purchase', 'sold',
            'damaged', 'faulty', 'replacement', 'exchange'
        ]
        
        # Criminal Law / Threat Keywords (High urgency)
        criminal_keywords = [
            'threat', 'false case', 'extortion', 'blackmail', 'dowry',
            'harassment', 'abuse', 'violence', 'rape case', 'police case',
            'fir', 'arrest', 'custody', 'bail', 'criminal'
        ]
        
        # Family Law Keywords
        family_keywords = [
            'marriage', 'divorce', 'husband', 'wife', 'child', 'custody',
            'maintenance', 'alimony', 'family', 'domestic', 'marital'
        ]
        
        # Property Law Keywords
        property_keywords = [
            'property', 'land', 'house', 'dispute', 'neighbor', 'boundary',
            'encroachment', 'possession', 'title', 'deed', 'sale', 'rent'
        ]
        
        # Check for matches and classify
        consumer_score = sum(1 for kw in consumer_keywords if kw in query_lower)
        criminal_score = sum(1 for kw in criminal_keywords if kw in query_lower)
        family_score = sum(1 for kw in family_keywords if kw in query_lower)
        property_score = sum(1 for kw in property_keywords if kw in query_lower)
        
        # Find the highest scoring category
        scores = {
            QueryType.CONSUMER_PROTECTION: consumer_score,
            QueryType.CRIMINAL_LAW: criminal_score,
            QueryType.FAMILY_LAW: family_score,
            QueryType.PROPERTY_LAW: property_score
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            query_type = max(scores, key=scores.get)
            confidence = min(0.9, max_score * 0.3)  # Cap at 0.9
        
        # Determine urgency
        if criminal_score > 0:
            urgency = UrgencyLevel.CRITICAL if criminal_score >= 2 else UrgencyLevel.HIGH
            requires_legal_counsel = True
        elif family_score >= 2:
            urgency = UrgencyLevel.MEDIUM
            requires_legal_counsel = True
        elif property_score >= 2:
            urgency = UrgencyLevel.MEDIUM
        elif consumer_score >= 1:
            urgency = UrgencyLevel.LOW
        
        # Extract keywords from query
        all_keywords = consumer_keywords + criminal_keywords + family_keywords + property_keywords
        keywords = [kw for kw in all_keywords if kw in query_lower]
        
        return QueryClassification(
            query_type=query_type,
            urgency=urgency,
            confidence=confidence,
            requires_legal_counsel=requires_legal_counsel,
            keywords=keywords,
            priority_score=urgency.value * confidence
        )


# Main execution example
if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        api_token = os.getenv('INDIAN_KANOON_API_TOKEN')
        if not api_token:
            print("Please set INDIAN_KANOON_API_TOKEN environment variable")
            return
        
        test_query = "I bought a defective phone, what should I do?"
        
        result = await agentic_legal_search(api_token, test_query)
        
        print("Agentic Legal Search Results:")
        print(f"Query: {test_query}")
        print(f"Classification: {result.get('classification', {})}")
        print(f"Search Results: {len(result.get('search_results', {}).get('results', []))} found")
        print(f"Recommendations: {len(result.get('recommendations', []))} provided")
    
    # Run example
    asyncio.run(main())
