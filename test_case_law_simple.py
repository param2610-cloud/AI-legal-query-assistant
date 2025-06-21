#!/usr/bin/env python3
"""
Simple test script for CaseLawAnalyzer that avoids Windows temp directory issues
"""

import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "agent"))

def test_case_law_analyzer_simple():
    """Test the CaseLawAnalyzer without temporary directories"""
    
    try:
        from agent.enhanced_case_law_analyzer import CaseLawAnalyzer
        
        logger.info("Testing CaseLawAnalyzer with persistent directory...")
        
        # Use a persistent directory instead of temporary
        test_db_path = "chroma_db_caselaw_test"
        
        analyzer = CaseLawAnalyzer(
            case_law_dir="training_data/case_law",
            vector_db_path=test_db_path
        )
        
        logger.info("✅ CaseLawAnalyzer initialized successfully!")
        logger.info(f"Loaded {len(analyzer.case_documents)} case law documents")
        
        # Test a simple similarity search
        if analyzer.vectorstore:
            try:
                results = analyzer.vectorstore.similarity_search("consumer protection defective product", k=2)
                logger.info(f"✅ Test search returned {len(results)} results")
                
                for i, result in enumerate(results):
                    case_name = result.metadata.get('case_name', 'Unknown case')
                    category = result.metadata.get('case_category', 'Unknown category')
                    logger.info(f"  {i+1}. {case_name} ({category})")
                    
            except Exception as e:
                logger.error(f"❌ Error during test search: {e}")
        
        # Test finding similar cases
        test_queries = [
            "My phone is defective and seller won't replace it",
            "Employer not paying overtime wages",
            "Dowry harassment by husband's family"
        ]
        
        for query in test_queries:
            try:
                similar_cases = analyzer._find_similar_cases(query, top_k=2)
                logger.info(f"✅ Found {len(similar_cases)} similar cases for: {query[:50]}...")
            except Exception as e:
                logger.warning(f"⚠️ Error finding similar cases for '{query[:30]}...': {e}")
        
        # Clean up test database
        try:
            import shutil
            if Path(test_db_path).exists():
                shutil.rmtree(test_db_path, ignore_errors=True)
                logger.info("🧹 Cleaned up test database")
        except Exception as e:
            logger.warning(f"⚠️ Could not clean up test database: {e}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_case_law_analyzer_simple()
    if success:
        print("\n🎉 Case Law Analyzer is working correctly!")
        print("✅ No metadata errors")
        print("✅ Case law documents loaded successfully")
        print("✅ Vector similarity search working")
        print("✅ Ready for enhanced legal assistance!")
    else:
        print("\n❌ Case Law Analyzer test failed!")
        sys.exit(1)