#!/usr/bin/env python3
"""
Enhanced Legal Assistant Test Script
===================================

This script tests the enhanced legal assistant with comprehensive case law analysis,
strategic guidance, and verdict pattern recognition.

Features:
1. Case law analysis with verdict patterns
2. Strategic recommendations based on precedents
3. Success/failure factor analysis
4. Indian Kanoon API integration for live search
5. Confidence scoring and risk assessment

Usage:
    python test_enhanced_legal_assistant.py
    python test_enhanced_legal_assistant.py --interactive
    python test_enhanced_legal_assistant.py --benchmark
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "agent"))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedLegalAssistantTester:
    """Comprehensive tester for the enhanced legal assistant"""
    
    def __init__(self):
        self.project_root = project_root
        self.assistant = None
        self.vectorstore = None
        self.analyzer = None
        
    def setup_assistant(self):
        """Setup the enhanced legal assistant"""
        
        try:
            # Import the enhanced components
            from agent.legal_assistant import SimpleLegalAssistant
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from train_case_law_system import AdvancedCaseLawAnalyzer
            
            # Initialize the basic assistant
            self.assistant = SimpleLegalAssistant()
            
            # Check if assistant initialization works
            if not self.assistant.initialize():
                logger.error("Failed to initialize basic legal assistant")
                return False
            
            # Setup case law database
            db_path = str(self.project_root / "chroma_db_caselaw")
            
            if not Path(db_path).exists():
                logger.error(f"Case law database not found at {db_path}")
                logger.info("Please run: python train_case_law_system.py")
                return False
            
            # Load case law vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            self.vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
            
            # Initialize advanced analyzer
            self.analyzer = AdvancedCaseLawAnalyzer(self.vectorstore)
            
            logger.info("✅ Enhanced legal assistant setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up assistant: {e}")
            return False
    
    async def run_comprehensive_test(self):
        """Run comprehensive test with various legal scenarios"""
        
        test_scenarios = {
            "Consumer Protection": [
                "I bought a defective smartphone online. The seller refuses to replace it even though it's under warranty. What are my legal options?",
                "My bank charged me unfair fees and won't refund them. How can I file a consumer complaint?",
                "I received a damaged product but the e-commerce company is not responding. What should I do?"
            ],
            "Employment Law": [
                "My employer is making me work 14 hours daily without overtime pay. Is this legal?",
                "I was fired without notice and they're not paying my salary. What are my rights?",
                "My workplace is unsafe and management ignores safety complaints. What legal action can I take?"
            ],
            "Family Law": [
                "My husband demands dowry and threatens me. How can I protect myself legally?",
                "We want to divorce by mutual consent. What is the legal process?",
                "My child custody case is pending. What factors do courts consider?"
            ],
            "Property Law": [
                "The builder is asking for extra charges beyond the agreed amount. Is this legal?",
                "My property documents are disputed by neighbors. How do I resolve this?",
                "I want to buy a property but the paperwork seems incomplete. What should I verify?"
            ],
            "Criminal Law": [
                "Someone filed a false police complaint against me. What can I do?",
                "I need to get bail for a family member. What is the process?",
                "How do I file an FIR if police are not cooperating?"
            ]
        }
        
        print("🏛️ COMPREHENSIVE LEGAL ASSISTANT TEST")
        print("=" * 60)
        
        for category, queries in test_scenarios.items():
            print(f"\n📂 **{category.upper()} SCENARIOS**")
            print("-" * 40)
            
            for i, query in enumerate(queries, 1):
                print(f"\n{i}. {query}")
                await self.test_single_query(query, category)
                print("\n" + "="*60)
    
    async def test_single_query(self, query: str, category: str = "General"):
        """Test a single query comprehensively"""
        
        try:
            print(f"\n🔍 **ANALYZING QUERY:** {query}")
            print("-" * 40)
            
            # 1. Basic Legal Response
            print("📄 **1. BASIC LEGAL GUIDANCE:**")
            basic_response = self.assistant.ask_question(query)
            if basic_response and basic_response.get('response'):
                response_text = basic_response['response']
                if isinstance(response_text, dict) and 'result' in response_text:
                    response_text = response_text['result']
                elif isinstance(response_text, dict):
                    response_text = str(response_text)
                    
                print(response_text[:500] + "..." if len(str(response_text)) > 500 else response_text)
            
            # 2. Case Law Analysis
            print(f"\n⚖️ **2. CASE LAW ANALYSIS:**")
            if self.analyzer and self.analyzer.llm:
                verdict_analysis = self.analyzer.analyze_verdict_judgment(query)
                
                total_cases = verdict_analysis.get('total_cases_found', 0)
                print(f"   📊 Cases Analyzed: {total_cases}")
                
                if verdict_analysis.get('verdict_patterns'):
                    print("   📈 Verdict Patterns:")
                    for pattern in verdict_analysis['verdict_patterns'][:3]:
                        print(f"      • {pattern}")
                
                if verdict_analysis.get('success_factors'):
                    print("   ✅ Success Factors:")
                    for factor in verdict_analysis['success_factors'][:2]:
                        print(f"      • {factor}")
                
                if verdict_analysis.get('failure_factors'):
                    print("   ❌ Common Failure Factors:")
                    for factor in verdict_analysis['failure_factors'][:2]:
                        print(f"      • {factor}")
            else:
                print("   ⚠️ Advanced case law analysis not available")
            
            # 3. Strategic Recommendations
            print(f"\n🎯 **3. STRATEGIC RECOMMENDATIONS:**")
            if self.analyzer and self.analyzer.llm:
                strategy = self.analyzer.generate_strategy_recommendations(query)
                if strategy.get('strategy_recommendation'):
                    # Extract key points from strategy
                    strategy_text = strategy['strategy_recommendation']
                    if hasattr(strategy_text, 'content'):
                        strategy_text = strategy_text.content
                    
                    lines = str(strategy_text).split('\n')
                    key_points = [line.strip() for line in lines if line.strip() and 
                                (line.strip().startswith('1.') or line.strip().startswith('2.') or 
                                 line.strip().startswith('3.') or line.strip().startswith('•') or
                                 line.strip().startswith('-'))][:5]
                    
                    for point in key_points:
                        print(f"   {point}")
                    
                    confidence = strategy.get('confidence', 0)
                    print(f"   📊 Confidence Score: {confidence:.2f}")
            else:
                print("   ⚠️ Strategic analysis not available")
            
            # 4. Similar Cases Search
            print(f"\n🔍 **4. SIMILAR CASES:**")
            if self.analyzer:
                similar_cases = await self.analyzer.search_similar_cases_online(query, max_results=3)
                if similar_cases.get('status') == 'success':
                    print("   ✅ Online case search successful")
                    online_results = similar_cases.get('online_results', {})
                    if online_results.get('search_results'):
                        print(f"   📁 Found {len(online_results['search_results'])} similar cases")
                elif similar_cases.get('status') == 'unavailable':
                    local_results = similar_cases.get('local_results', {})
                    local_count = local_results.get('total_found', 0)
                    print(f"   📁 Found {local_count} similar cases locally")
                    
                    if local_results.get('cases'):
                        print("   📄 Top Similar Cases:")
                        for case in local_results['cases'][:2]:
                            case_name = case.get('case_name', 'Unknown')
                            court = case.get('court', 'Unknown')
                            year = case.get('year', 'Unknown')
                            print(f"      • {case_name} ({court}, {year})")
            
            # 5. Risk Assessment
            print(f"\n⚠️ **5. RISK ASSESSMENT:**")
            print("   • Legal complexity: Medium to High")
            print("   • Recommended action: Consult qualified lawyer")
            print("   • Time sensitivity: Varies by case")
            print("   • Success probability: Depends on evidence and precedents")
            
        except Exception as e:
            logger.error(f"Error testing query: {e}")
            print(f"   ❌ Error: {e}")
    
    async def interactive_mode(self):
        """Interactive mode for testing"""
        
        print("🤖 ENHANCED LEGAL ASSISTANT - INTERACTIVE MODE")
        print("=" * 50)
        print("Ask legal questions and get comprehensive analysis!")
        print("Type 'quit' to exit, 'help' for sample questions")
        
        sample_questions = [
            "Consumer complaint process for defective products",
            "Employment rights for overtime pay",
            "Dowry harassment legal protection",
            "Property document verification process",
            "FIR filing procedure and rights"
        ]
        
        while True:
            print("\n" + "-" * 50)
            user_input = input("\n💬 Your legal question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using the Enhanced Legal Assistant!")
                break
            elif user_input.lower() in ['help', 'h']:
                print("\n💡 Sample questions you can ask:")
                for i, q in enumerate(sample_questions, 1):
                    print(f"{i}. {q}")
                continue
            elif not user_input:
                print("Please enter a legal question or 'help' for examples.")
                continue
            
            await self.test_single_query(user_input)
    
    def run_benchmark(self):
        """Run benchmark tests"""
        
        print("🏁 BENCHMARK TESTS")
        print("=" * 30)
        
        # Test database connectivity
        print("1. Testing database connectivity...")
        if self.vectorstore:
            try:
                # Test retrieval
                test_query = "consumer protection"
                docs = self.vectorstore.similarity_search(test_query, k=3)
                print(f"   ✅ Retrieved {len(docs)} documents")
            except Exception as e:
                print(f"   ❌ Database error: {e}")
        
        # Test analyzer functionality
        print("2. Testing case law analyzer...")
        if self.analyzer:
            try:
                if self.analyzer.llm:
                    print("   ✅ LLM model loaded")
                else:
                    print("   ⚠️ LLM model not available")
                
                if self.analyzer.indian_kanoon_client:
                    print("   ✅ Indian Kanoon client available")
                else:
                    print("   ⚠️ Indian Kanoon client not available")
                    
            except Exception as e:
                print(f"   ❌ Analyzer error: {e}")
        
        # Test basic assistant
        print("3. Testing basic legal assistant...")
        if self.assistant:
            try:
                test_response = self.assistant.ask_question("What is consumer protection?")
                if test_response and test_response.get('response'):
                    print("   ✅ Basic assistant responding")
                else:
                    print("   ❌ Basic assistant not responding")
            except Exception as e:
                print(f"   ❌ Assistant error: {e}")
        
        print("\n🎯 Benchmark complete!")

async def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Enhanced Legal Assistant Tester")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--benchmark", "-b", action="store_true",
                       help="Run benchmark tests")
    parser.add_argument("--comprehensive", "-c", action="store_true",
                       help="Run comprehensive test suite")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = EnhancedLegalAssistantTester()
    
    print("🚀 Setting up Enhanced Legal Assistant...")
    if not tester.setup_assistant():
        print("❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    
    # Run requested mode
    if args.interactive:
        await tester.interactive_mode()
    elif args.benchmark:
        tester.run_benchmark()
    elif args.comprehensive:
        await tester.run_comprehensive_test()
    else:
        # Default: run a quick test
        print("\n🧪 Quick Test Mode")
        print("Use --interactive for interactive mode")
        print("Use --comprehensive for full test suite")
        print("Use --benchmark for system checks")
        
        # Quick test
        await tester.test_single_query(
            "My employer is not paying overtime. What are my legal rights?",
            "Employment Law"
        )

if __name__ == "__main__":
    asyncio.run(main())
