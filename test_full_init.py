#!/usr/bin/env python3
"""
Test the full SimpleLegalAssistant initialization to reproduce the error
"""

import sys
from pathlib import Path

# Add the agent directory to path
agent_dir = Path(__file__).parent / "agent"
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

def test_full_initialization():
    """Test the full SimpleLegalAssistant initialization"""
    try:
        print("=== Testing Full SimpleLegalAssistant Initialization ===")
        
        print("1. Importing SimpleLegalAssistant...")
        from legal_assistant import SimpleLegalAssistant
        
        print("2. Creating SimpleLegalAssistant instance...")
        assistant = SimpleLegalAssistant()
        
        print("3. Checking RAG system...")
        print(f"RAG System type: {type(assistant.rag_system)}")
        print(f"RAG System methods: {[m for m in dir(assistant.rag_system) if not m.startswith('_')]}")
        
        print("4. Checking if create_legal_qa_chain exists on rag_system...")
        if hasattr(assistant.rag_system, 'create_legal_qa_chain'):
            print("✅ create_legal_qa_chain method exists on rag_system!")
        else:
            print("❌ create_legal_qa_chain method NOT found on rag_system!")
            return False
        
        print("5. Attempting to initialize...")
        success = assistant.initialize()
        
        if success:
            print("✅ Initialization completed successfully!")
            return True
        else:
            print("❌ Initialization failed!")
            return False
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_initialization()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed - check errors above")
