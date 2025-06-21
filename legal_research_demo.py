"""
Indian Kanoon API Integration Example
===================================

This script demonstrates how to integrate Indian Kanoon API with the AI Legal Assistant
to provide enhanced legal research capabilities with live case law data.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from indian_kanoon_api import IndianKanoonAPI
from indian_kanoon_config import IndianKanoonConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalResearchAssistant:
    """
    Enhanced legal assistant with Indian Kanoon API integration.
    Provides live case law search and document retrieval capabilities.
    """
    
    def __init__(self, budget_limit: float = 500.0):
        """Initialize the legal research assistant."""
        self.budget_limit = budget_limit
        self.api = None
        self._initialize_api()
    
    def _initialize_api(self):
        """Initialize Indian Kanoon API client."""
        try:
            # Validate configuration
            if not IndianKanoonConfig.validate_config():
                logger.warning("Indian Kanoon API not properly configured. Limited functionality available.")
                return
            
            # Initialize API client based on available authentication
            if IndianKanoonConfig.API_TOKEN:
                self.api = IndianKanoonAPI(
                    api_token=IndianKanoonConfig.API_TOKEN,
                    budget_limit=self.budget_limit
                )
                logger.info("✅ Indian Kanoon API initialized with token authentication")
            
            elif IndianKanoonConfig.CUSTOMER_EMAIL and IndianKanoonConfig.PRIVATE_KEY_PATH.exists():
                self.api = IndianKanoonAPI(
                    customer_email=IndianKanoonConfig.CUSTOMER_EMAIL,
                    private_key_path=str(IndianKanoonConfig.PRIVATE_KEY_PATH),
                    budget_limit=self.budget_limit
                )
                logger.info("✅ Indian Kanoon API initialized with crypto authentication")
            
        except Exception as e:
            logger.error(f"Failed to initialize Indian Kanoon API: {e}")
            self.api = None
    
    def search_consumer_cases(self, issue_keywords: List[str], max_results: int = 5) -> List[Dict]:
        """
        Search for consumer protection related cases.
        
        Args:
            issue_keywords: Keywords describing the consumer issue
            max_results: Maximum number of cases to return
            
        Returns:
            List of relevant consumer cases
        """
        if not self.api:
            logger.warning("Indian Kanoon API not available")
            return []
        
        try:
            # Build consumer-focused query
            consumer_terms = ['consumer', 'protection']
            all_keywords = consumer_terms + issue_keywords
            
            cases = self.api.search_legal_cases(
                case_keywords=all_keywords,
                court_type='supreme_court',  # Start with Supreme Court cases
                max_results=max_results
            )
            
            # Enhance case information
            enhanced_cases = []
            for case in cases:
                enhanced_case = {
                    'title': case['title'],
                    'court': case['court'],
                    'date': case['date'],
                    'summary': case['summary'],
                    'relevance_score': self._calculate_relevance(case, issue_keywords),
                    'doc_id': case['doc_id'],
                    'key_points': self._extract_key_points(case)
                }
                enhanced_cases.append(enhanced_case)
            
            # Sort by relevance
            enhanced_cases.sort(key=lambda x: x['relevance_score'], reverse=True)
            return enhanced_cases
            
        except Exception as e:
            logger.error(f"Consumer case search failed: {e}")
            return []
    
    def search_employment_cases(self, issue_keywords: List[str], max_results: int = 5) -> List[Dict]:
        """Search for employment law related cases."""
        if not self.api:
            return []
        
        try:
            employment_terms = ['employment', 'labor', 'labour', 'worker', 'employee']
            all_keywords = employment_terms + issue_keywords
            
            cases = self.api.search_legal_cases(
                case_keywords=all_keywords,
                court_type='all_judgments',
                max_results=max_results
            )
            
            return self._enhance_cases(cases, issue_keywords)
            
        except Exception as e:
            logger.error(f"Employment case search failed: {e}")
            return []
    
    def search_family_law_cases(self, issue_keywords: List[str], max_results: int = 5) -> List[Dict]:
        """Search for family law related cases."""
        if not self.api:
            return []
        
        try:
            family_terms = ['marriage', 'divorce', 'family', 'matrimonial', 'custody']
            all_keywords = family_terms + issue_keywords
            
            cases = self.api.search_legal_cases(
                case_keywords=all_keywords,
                court_type='all_judgments',
                max_results=max_results
            )
            
            return self._enhance_cases(cases, issue_keywords)
            
        except Exception as e:
            logger.error(f"Family law case search failed: {e}")
            return []
    
    def get_case_details(self, doc_id: str) -> Optional[Dict]:
        """Get detailed information about a specific case."""
        if not self.api:
            return None
        
        try:
            document = self.api.get_document(doc_id, max_cites=10, max_cited_by=5)
            
            # Extract key information
            case_details = {
                'doc_id': doc_id,
                'title': document.get('title', ''),
                'court': document.get('court', ''),
                'date': document.get('date', ''),
                'content': document.get('doc', ''),
                'citations': document.get('citeList', []),
                'cited_by': document.get('citedbyList', []),
                'judges': document.get('author', ''),
                'summary': self._generate_case_summary(document)
            }
            
            return case_details
            
        except Exception as e:
            logger.error(f"Failed to get case details for {doc_id}: {e}")
            return None
    
    def _calculate_relevance(self, case: Dict, keywords: List[str]) -> float:
        """Calculate relevance score for a case based on keywords."""
        score = 0.0
        case_text = f"{case.get('title', '')} {case.get('summary', '')}".lower()
        
        for keyword in keywords:
            if keyword.lower() in case_text:
                score += 1.0
        
        # Bonus for newer cases
        if case.get('date'):
            try:
                year = int(case['date'].split('-')[-1])
                if year >= 2020:
                    score += 0.5
                elif year >= 2010:
                    score += 0.2
            except:
                pass
        
        return score
    
    def _enhance_cases(self, cases: List[Dict], keywords: List[str]) -> List[Dict]:
        """Enhance case information with relevance scoring."""
        enhanced_cases = []
        for case in cases:
            enhanced_case = {
                'title': case['title'],
                'court': case['court'],
                'date': case['date'],
                'summary': case['summary'],
                'relevance_score': self._calculate_relevance(case, keywords),
                'doc_id': case['doc_id'],
                'key_points': self._extract_key_points(case)
            }
            enhanced_cases.append(enhanced_case)
        
        enhanced_cases.sort(key=lambda x: x['relevance_score'], reverse=True)
        return enhanced_cases
    
    def _extract_key_points(self, case: Dict) -> List[str]:
        """Extract key points from case summary."""
        summary = case.get('summary', '')
        if not summary:
            return []
        
        # Simple key point extraction (can be enhanced with NLP)
        sentences = summary.split('.')
        key_points = []
        
        keywords = ['held', 'decided', 'ruled', 'ordered', 'directed', 'observed']
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in keywords):
                key_points.append(sentence)
        
        return key_points[:3]  # Return top 3 key points
    
    def _generate_case_summary(self, document: Dict) -> str:
        """Generate a concise summary of the case."""
        content = document.get('doc', '')
        if len(content) < 500:
            return content
        
        # Extract first few paragraphs
        paragraphs = content.split('\n\n')
        summary_parts = []
        char_count = 0
        
        for paragraph in paragraphs:
            if char_count + len(paragraph) > 1000:
                break
            summary_parts.append(paragraph)
            char_count += len(paragraph)
        
        return '\n\n'.join(summary_parts) + "..."
    
    def get_budget_status(self) -> Dict:
        """Get current API budget status."""
        if self.api:
            return self.api.get_budget_status()
        return {'budget_limit': 0, 'current_spending': 0, 'remaining_budget': 0}
    
    def save_usage_report(self, filename: str = None):
        """Save API usage report."""
        if self.api:
            if not filename:
                filename = f"indian_kanoon_usage_{int(time.time())}.json"
            self.api.save_request_history(filename)
            logger.info(f"Usage report saved to {filename}")

def demo_legal_research():
    """Demonstrate the legal research capabilities."""
    print("🔍 AI Legal Research Assistant - Indian Kanoon Integration Demo")
    print("=" * 70)
    
    # Initialize assistant
    assistant = LegalResearchAssistant(budget_limit=50.0)  # Small budget for demo
    
    if not assistant.api:
        print("❌ Indian Kanoon API not available. Please configure authentication.")
        return
    
    # Demo 1: Consumer Protection Case Search
    print("\n📱 Demo 1: Consumer Protection Cases")
    print("-" * 40)
    consumer_cases = assistant.search_consumer_cases(
        issue_keywords=['defective', 'product', 'refund'],
        max_results=3
    )
    
    for i, case in enumerate(consumer_cases, 1):
        print(f"{i}. {case['title'][:80]}...")
        print(f"   Court: {case['court']}")
        print(f"   Date: {case['date']}")
        print(f"   Relevance: {case['relevance_score']:.1f}")
        print()
    
    # Demo 2: Employment Law Case Search
    print("\n💼 Demo 2: Employment Law Cases")
    print("-" * 40)
    employment_cases = assistant.search_employment_cases(
        issue_keywords=['overtime', 'working', 'hours'],
        max_results=3
    )
    
    for i, case in enumerate(employment_cases, 1):
        print(f"{i}. {case['title'][:80]}...")
        print(f"   Court: {case['court']}")
        print(f"   Date: {case['date']}")
        print()
    
    # Budget Status
    print("\n💰 Budget Status")
    print("-" * 40)
    budget = assistant.get_budget_status()
    print(f"Spent: Rs {budget['current_spending']:.2f}")
    print(f"Remaining: Rs {budget['remaining_budget']:.2f}")
    print(f"Usage: {budget['usage_percentage']:.1f}%")
    
    # Save usage report
    assistant.save_usage_report()
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    import time
    demo_legal_research()
