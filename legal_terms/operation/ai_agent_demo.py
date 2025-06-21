#!/usr/bin/env python3
"""
AI Agent Integration Demo for Legal Terms

This script demonstrates how an AI agent can use the extracted legal terms
for various operations like term lookup, semantic search, and categorization.
"""

import json
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict

class LegalTermsAIAgent:
    """
    Demo AI Agent that uses extracted legal terms for various operations
    """
    
    def __init__(self, terms_file: str = "legal_terms_structured.json"):
        self.terms_file = terms_file
        self.data = None
        self.load_terms()
    
    def load_terms(self) -> bool:
        """Load legal terms data"""
        try:
            with open(self.terms_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✅ Loaded {self.data['metadata']['total_terms']} legal terms")
            return True
        except Exception as e:
            print(f"❌ Error loading terms: {e}")
            return False
    
    def quick_lookup(self, term: str) -> Optional[str]:
        """Quick definition lookup for a term"""
        if not self.data:
            return None
        
        # Try exact match first
        term_lower = term.lower()
        if term_lower in self.data['term_definitions']:
            return self.data['term_definitions'][term_lower]
        
        # Try variant matching
        if term_lower in self.data['term_variants']:
            actual_term = self.data['term_variants'][term_lower]
            return self.data['term_definitions'].get(actual_term.lower())
        
        return None
    
    def fuzzy_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fuzzy search for terms matching the query"""
        if not self.data:
            return []
        
        query_lower = query.lower()
        matches = []
        
        for term_data in self.data['terms']:
            term = term_data['term']
            definition = term_data['definition']
            
            # Calculate relevance score
            score = 0
            
            # Exact term match gets highest score
            if query_lower == term.lower():
                score += 100
            
            # Term contains query
            elif query_lower in term.lower():
                score += 80
            
            # Query contains term
            elif term.lower() in query_lower:
                score += 60
            
            # Definition contains query
            elif query_lower in definition.lower():
                score += 40
            
            # Keywords match
            for keyword in term_data.get('keywords', []):
                if keyword in query_lower:
                    score += 20
            
            # Variants match
            for variant in term_data.get('variants', []):
                if query_lower in variant.lower() or variant.lower() in query_lower:
                    score += 30
            
            if score > 0:
                matches.append({
                    'term': term,
                    'definition': definition,
                    'categories': term_data.get('categories', []),
                    'score': score
                })
        
        # Sort by score and return top matches
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:limit]
    
    def get_terms_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all terms belonging to a specific category"""
        if not self.data:
            return []
        
        return self.data['terms_by_category'].get(category, [])
    
    def search_by_keywords(self, keywords: List[str]) -> List[str]:
        """Search terms by keywords"""
        if not self.data:
            return []
        
        relevant_terms = set()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self.data['keywords_index']:
                relevant_terms.update(self.data['keywords_index'][keyword_lower])
        
        return list(relevant_terms)
    
    def explain_legal_concept(self, concept: str) -> Dict[str, Any]:
        """Provide comprehensive explanation of a legal concept"""
        # First try direct lookup
        definition = self.quick_lookup(concept)
        
        if definition:
            # Get related terms
            related_matches = self.fuzzy_search(concept, limit=3)
            related_terms = [match for match in related_matches if match['term'].lower() != concept.lower()]
            
            # Get category information
            term_data = None
            for term in self.data['terms']:
                if term['term'].lower() == concept.lower():
                    term_data = term
                    break
            
            return {
                'term': concept,
                'definition': definition,
                'categories': term_data.get('categories', []) if term_data else [],
                'keywords': term_data.get('keywords', []) if term_data else [],
                'related_terms': related_terms[:3],
                'found': True
            }
        else:
            # Try fuzzy search if direct lookup fails
            matches = self.fuzzy_search(concept, limit=5)
            return {
                'term': concept,
                'definition': None,
                'possible_matches': matches,
                'found': False
            }
    
    def analyze_legal_text(self, text: str) -> Dict[str, Any]:
        """Analyze text and identify legal terms within it"""
        if not self.data:
            return {}
        
        found_terms = []
        text_lower = text.lower()
        
        # Check for exact term matches
        for term_data in self.data['terms']:
            term = term_data['term']
            if term.lower() in text_lower:
                found_terms.append({
                    'term': term,
                    'definition': term_data['definition'],
                    'categories': term_data['categories'],
                    'position': text_lower.find(term.lower())
                })
            
            # Also check variants
            for variant in term_data.get('variants', []):
                if variant.lower() in text_lower and variant.lower() != term.lower():
                    found_terms.append({
                        'term': f"{term} (as '{variant}')",
                        'definition': term_data['definition'],
                        'categories': term_data['categories'],
                        'position': text_lower.find(variant.lower())
                    })
        
        # Remove duplicates and sort by position
        unique_terms = []
        seen_terms = set()
        
        for term_info in sorted(found_terms, key=lambda x: x['position']):
            term_key = term_info['term'].split(' (as ')[0]  # Get base term
            if term_key not in seen_terms:
                unique_terms.append(term_info)
                seen_terms.add(term_key)
        
        # Categorize found terms
        category_counts = defaultdict(int)
        for term_info in unique_terms:
            for category in term_info['categories']:
                category_counts[category] += 1
        
        return {
            'text_length': len(text),
            'found_terms': unique_terms,
            'total_terms_found': len(unique_terms),
            'categories_present': dict(category_counts),
            'legal_areas': list(category_counts.keys())
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the legal terms database"""
        if not self.data:
            return {}
        
        return {
            'total_terms': self.data['metadata']['total_terms'],
            'categories': self.data['metadata']['categories'],
            'extraction_date': self.data['metadata']['extraction_date'],
            'terms_by_category_count': {
                category: len(terms) 
                for category, terms in self.data['terms_by_category'].items()
            },
            'average_definition_length': sum(
                len(term['definition']) for term in self.data['terms']
            ) / len(self.data['terms'])
        }

def demo_ai_agent():
    """Demonstrate AI agent capabilities"""
    print("=" * 80)
    print("AI LEGAL AGENT DEMONSTRATION")
    print("=" * 80)
    
    agent = LegalTermsAIAgent()
    
    if not agent.data:
        print("❌ Could not load legal terms data")
        return
    
    # Demo 1: Quick lookup
    print("\n🔍 DEMO 1: Quick Term Lookup")
    print("-" * 40)
    test_terms = ["bail", "murder", "affidavit", "habeas corpus"]
    
    for term in test_terms:
        definition = agent.quick_lookup(term)
        if definition:
            print(f"✅ {term}: {definition[:80]}...")
        else:
            print(f"❌ {term}: Not found")
    
    # Demo 2: Fuzzy search
    print("\n🔍 DEMO 2: Fuzzy Search")
    print("-" * 40)
    search_queries = ["criminal", "court procedure", "evidence"]
    
    for query in search_queries:
        matches = agent.fuzzy_search(query, limit=3)
        print(f"\nQuery: '{query}'")
        for i, match in enumerate(matches):
            print(f"  {i+1}. {match['term']} (Score: {match['score']})")
            print(f"     {match['definition'][:60]}...")
    
    # Demo 3: Category analysis
    print("\n🏷️  DEMO 3: Category Analysis")
    print("-" * 40)
    categories = agent.data['metadata']['categories'][:3]  # Show top 3
    
    for category in categories:
        terms = agent.get_terms_by_category(category)
        print(f"{category}: {len(terms)} terms")
        if terms:
            sample_terms = [term['term'] for term in terms[:3]]
            print(f"  Examples: {', '.join(sample_terms)}")
    
    # Demo 4: Legal text analysis
    print("\n📝 DEMO 4: Legal Text Analysis")
    print("-" * 40)
    sample_text = """
    The accused person was arrested without a warrant and taken into police custody.
    The judge issued a bail order after reviewing the evidence and considering the 
    charges. The defendant's advocate filed an appeal challenging the judgment.
    """
    
    analysis = agent.analyze_legal_text(sample_text)
    print(f"Text analyzed: {analysis['text_length']} characters")
    print(f"Legal terms found: {analysis['total_terms_found']}")
    print(f"Legal areas involved: {', '.join(analysis['legal_areas'])}")
    
    print("\nFound terms:")
    for term_info in analysis['found_terms'][:5]:  # Show first 5
        print(f"  • {term_info['term']}")
        print(f"    Categories: {', '.join(term_info['categories'])}")
    
    # Demo 5: Statistics
    print("\n📊 DEMO 5: Database Statistics")
    print("-" * 40)
    stats = agent.get_statistics()
    print(f"Total terms: {stats['total_terms']}")
    print(f"Categories: {len(stats['categories'])}")
    print(f"Average definition length: {stats['average_definition_length']:.1f} characters")
    
    print("\nTop categories by term count:")
    category_counts = sorted(
        stats['terms_by_category_count'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]
    
    for category, count in category_counts:
        print(f"  {category}: {count} terms")
    
    print(f"\n✅ AI Agent demonstration completed!")
    print(f"   The legal terms database is ready for integration with your AI system.")

if __name__ == "__main__":
    demo_ai_agent()
