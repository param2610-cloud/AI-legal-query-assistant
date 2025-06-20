"""
Test Script for Modern Legal Assistant
====================================

This script demonstrates how to use the modern agentic RAG system
for legal assistance without any paid API keys.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent.modern_legal_assistant import ModernLegalAssistant

def test_legal_assistant():
    """Test the modern legal assistant"""
    
    print("🚀 Testing Modern Legal Assistant (Agentic RAG)")
    print("=" * 50)
    
    # Initialize assistant
    assistant = ModernLegalAssistant(
        model_name="llama3.2:3b",  # Free local model via Ollama
        data_dir="indian_code/acts"
    )
    
    # Initialize the system
    print("📚 Loading legal documents and setting up RAG system...")
    if not assistant.initialize():
        print("❌ Failed to initialize the assistant")
        return False
    
    print("✅ System initialized successfully!")
    
    # Test questions
    test_questions = [
        "What are my rights as a consumer in India?",
        "My employer is not paying minimum wage. What legal action can I take?",
        "What are the child labor laws in India?",
        "How do I file a case in civil court?",
        "What is the procedure for marriage registration?"
    ]
    
    print(f"\n🧪 Testing with {len(test_questions)} sample questions:")
    print("-" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("   Processing...")
        
        result = assistant.ask_question(question)
        
        if result["status"] == "success":
            print(f"   ✅ Answer: {result['response'][:200]}...")
        elif result["status"] == "success_fallback":
            print(f"   ⚠️  Answer (fallback): {result['response'][:200]}...")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        
        print("   " + "-" * 40)
    
    # Test relevant acts finder
    print(f"\n🔍 Testing relevant acts finder:")
    relevant_acts = assistant.get_relevant_acts("consumer protection")
    print(f"   Found {len(relevant_acts)} relevant acts for 'consumer protection'")
    
    for act in relevant_acts[:2]:  # Show first 2
        print(f"   - {act['act_name']}: {len(act['relevant_sections'])} sections")
    
    print("\n🎉 Testing completed!")
    return True

def demo_interactive_mode():
    """Interactive demo mode"""
    
    print("\n🎯 Interactive Demo Mode")
    print("=" * 50)
    
    assistant = ModernLegalAssistant()
    
    if not assistant.initialize():
        print("❌ Failed to initialize the assistant")
        return
    
    print("✅ Ready! Ask your legal questions (type 'quit' to exit)")
    
    while True:
        print("\n" + "-" * 30)
        question = input("🤔 Your legal question: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question.strip():
            continue
        
        print("🔍 Processing your question...")
        result = assistant.ask_question(question)
        
        if result["status"] == "success":
            print(f"\n📋 Legal Guidance:\n{result['response']}")
        elif result["status"] == "success_fallback": 
            print(f"\n📋 Legal Guidance (via fallback):\n{result['response']}")
        else:
            print(f"\n❌ Error: {result.get('error', 'Unable to process question')}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Modern Legal Assistant")
    parser.add_argument(
        "--mode", 
        choices=["test", "interactive"], 
        default="test",
        help="Run mode: test (automated) or interactive"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "test":
            test_legal_assistant()
        else:
            demo_interactive_mode()
            
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
