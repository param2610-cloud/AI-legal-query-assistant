#!/usr/bin/env python3
"""
Test script to verify the case law analyzer metadata fix
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "agent"))

def test_case_law_analyzer():
    """Test the enhanced case law analyzer with corrected metadata"""
    
    try:
        from agent.enhanced_case_law_analyzer import CaseLawAnalyzer
        
        logger.info("Testing Enhanced Case Law Analyzer...")
        
        # Use a different database path to avoid conflicts
        analyzer = CaseLawAnalyzer(
            case_law_dir="training_data/case_law",
            vector_db_path="chroma_db_caselaw_test"
        )
        
        logger.info("Case Law Analyzer initialized successfully!")
        logger.info(f"Loaded {len(analyzer.case_documents)} case law documents")
        
        # Test a simple query
        if analyzer.vectorstore:
            try:
                results = analyzer.vectorstore.similarity_search("consumer protection", k=3)
                logger.info(f"Test search returned {len(results)} results")
                
                for i, result in enumerate(results):
                    logger.info(f"Result {i+1}: {result.metadata.get('case_name', 'Unknown case')}")
                    
            except Exception as e:
                logger.error(f"Error during test search: {e}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_case_law_analyzer()
    if success:
        logger.info("✅ Case Law Analyzer test completed successfully!")
    else:
        logger.error("❌ Case Law Analyzer test failed!")
        sys.exit(1)
