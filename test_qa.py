#!/usr/bin/env python3
"""
Test the legal assistant with a sample question
"""

import sys
from pathlib import Path

# Add the agent directory to path
agent_dir = Path(__file__).parent / "agent"
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

def test_question_answering():
    """Test asking a legal question"""
    try:
        print("=== Testing Question Answering ===")
        
        from legal_assistant import SimpleLegalAssistant
        
        print("Creating and initializing assistant...")
        assistant = SimpleLegalAssistant()
        
        if not assistant.initialize():
            print("❌ Failed to initialize assistant")
            return False
        
        print("✅ Assistant initialized successfully!")
        
        # Test a simple legal question
        test_question = "What are the basic rights of consumers in India?"
        
        print(f"\nAsking question: {test_question}")
        
        result = assistant.ask_question(test_question)
        
        if result.get("status") == "success":
            print("✅ Question answered successfully!")
            print(f"Response length: {len(result.get('response', ''))}")
            print(f"Legal terms found: {result.get('terms_count', 0)}")
            
            # Print first 200 characters of response
            response = result.get('response', '')
            if response:
                print(f"\nResponse preview: {response[:200]}...")
            
            return True
        else:
            print(f"❌ Question answering failed: {result.get('error', 'Unknown error')}")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_question_answering()
    if success:
        print("\n🎉 Question answering test passed!")
    else:
        print("\n💥 Question answering test failed!")
