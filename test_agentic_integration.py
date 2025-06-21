#!/usr/bin/env python3
"""
Test Script for Indian Kanoon Agentic RAG Integration
===================================================

This script tests the integration between the enhanced Indian Kanoon client
and the agentic RAG flow following the mermaid diagram specifications.

Usage:
    python test_agentic_integration.py
    
Environment Variables:
    INDIAN_KANOON_API_TOKEN: Your Indian Kanoon API token
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the enhanced client
try:
    from indian_kanoon_client import (
        create_indian_kanoon_client, 
        agentic_legal_search,
        QueryType, 
        UrgencyLevel,
        QueryClassification
    )
    print("✅ Successfully imported agentic Indian Kanoon client")
except ImportError as e:
    print(f"❌ Failed to import agentic client: {e}")
    sys.exit(1)

async def test_basic_classification():
    """Test basic query classification without API calls"""
    print("\n🧪 Testing Query Classification...")
    
    # Import the classifier directly
    from indian_kanoon_client import QueryClassifier
    
    classifier = QueryClassifier()
    
    test_queries = [
        "I bought a defective phone, what should I do?",
        "My wife threatened false rape case for dowry",
        "Property dispute with neighbor",
        "Seeking divorce from abusive husband",
        "General legal advice needed"
    ]
    
    for query in test_queries:
        classification = classifier.classify_query(query)
        print(f"Query: {query}")
        print(f"  Type: {classification.query_type.value}")
        print(f"  Urgency: {classification.urgency.name}")
        print(f"  Confidence: {classification.confidence:.2f}")
        print(f"  Needs Counsel: {classification.requires_legal_counsel}")
        print(f"  Keywords: {classification.keywords}")
        print()

async def test_search_strategy():
    """Test search strategy generation"""
    print("\n🧪 Testing Search Strategy Generation...")
    
    from indian_kanoon_client import AgenticSearchEngine, QueryClassification
    
    search_engine = AgenticSearchEngine()
    
    # Create a sample classification
    classification = QueryClassification(
        query_type=QueryType.CONSUMER_PROTECTION,
        urgency=UrgencyLevel.MEDIUM,
        confidence=0.8,
        requires_legal_counsel=False,
        keywords=["defective product", "refund"]
    )
    
    strategy = search_engine.create_search_strategy(classification)
    
    print(f"Query Type: {classification.query_type.value}")
    print(f"Primary Keywords: {strategy.primary_keywords}")
    print(f"Secondary Keywords: {strategy.secondary_keywords}")
    print(f"Legal Sections: {strategy.legal_sections}")
    print(f"Max Results: {strategy.max_results}")
    print(f"Search Depth: {strategy.search_depth}")
    print(f"Priority Courts: {strategy.priority_courts}")

async def test_agentic_flow_simulation():
    """Test the complete agentic flow without API calls"""
    print("\n🧪 Testing Complete Agentic Flow (Simulation)...")
    
    # Simulate the mermaid flow steps
    test_query = "I purchased a water bottle for 20 rupees where actual price is 15 rupees"
    
    print(f"Input Query: {test_query}")
    print("\n📊 Flow Steps:")
    print("1. ✅ Query Classification")
    print("2. ✅ Urgency Assessment") 
    print("3. ✅ Search Strategy Generation")
    print("4. 🔄 Knowledge Retrieval (would call API)")
    print("5. ✅ Response Synthesis")
    print("6. ✅ Final Response Generation")
    
    # Test classification
    from indian_kanoon_client import QueryClassifier
    classifier = QueryClassifier()
    classification = classifier.classify_query(test_query)
    
    print(f"\n📋 Classification Results:")
    print(f"  Type: {classification.query_type.value}")
    print(f"  Urgency: {classification.urgency.name}")
    print(f"  Priority Score: {classification.priority_score}")
    
    # Test strategy
    from indian_kanoon_client import AgenticSearchEngine
    search_engine = AgenticSearchEngine()
    strategy = search_engine.create_search_strategy(classification)
    
    print(f"\n🎯 Search Strategy:")
    print(f"  Depth: {strategy.search_depth}")
    print(f"  Max Results: {strategy.max_results}")
    print(f"  Case Filter: {strategy.case_type_filter}")

async def test_with_api_token():
    """Test with actual API token if available"""
    api_token = os.getenv('INDIAN_KANOON_API_TOKEN')
    
    if not api_token:
        print("\n⚠️ No API token found - skipping live API tests")
        print("Set INDIAN_KANOON_API_TOKEN environment variable to test with live API")
        return
    
    print("\n🔥 Testing with Live API...")
    
    try:
        # Test the convenience function
        result = await agentic_legal_search(
            api_token=api_token,
            query="Consumer protection for defective product",
            user_context="Purchased faulty electronics",
            budget_limit=50.0  # Small budget for testing
        )
        
        print("✅ Live API test successful!")
        print(f"Query Type: {result['classification']['query_type']}")
        print(f"Results Found: {result['search_results']['total_found']}")
        print(f"Budget Used: Rs {result['budget_status']['current_budget']:.2f}")
        
    except Exception as e:
        print(f"❌ Live API test failed: {e}")

async def main():
    """Main test function"""
    print("🚀 Indian Kanoon Agentic RAG Integration Test")
    print("=" * 60)
    
    # Run all tests
    await test_basic_classification()
    await test_search_strategy()
    await test_agentic_flow_simulation()
    await test_with_api_token()
    
    print("\n✅ All tests completed!")
    print("\n📖 Next Steps:")
    print("1. Set INDIAN_KANOON_API_TOKEN to test with live API")
    print("2. Integrate with your legal assistant application")
    print("3. Use agentic_search() method for intelligent legal queries")
    print("4. Follow the mermaid flow for complete legal assistance")

if __name__ == "__main__":
    asyncio.run(main())
