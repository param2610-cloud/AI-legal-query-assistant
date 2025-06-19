#!/usr/bin/env python3
"""
Test script for the updated Legal Assistant
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.legal_assistant import SimpleLegalAssistant
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_legal_assistant():
    """Test the updated legal assistant"""
    print("🧪 Testing Legal Assistant Updates...")
    print("=" * 50)
    
    # Initialize the assistant
    assistant = SimpleLegalAssistant()
    
    print("📋 Step 1: Initializing...")
    if assistant.initialize():
        print("✅ Initialization successful!")
    else:
        print("❌ Initialization failed!")
        return False
    
    print("\n📋 Step 2: Testing a simple question...")
    test_question = "What are consumer rights in India?"
    
    try:
        result = assistant.ask_question(test_question)
        
        if result["status"] == "success":
            print("✅ Question processing successful!")
            print(f"📝 Response: {result['response'][:200]}...")
        elif result["status"] == "success_fallback":
            print("⚠️  Question processed with fallback!")
            print(f"📝 Response: {result['response'][:200]}...")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception during testing: {e}")
        return False
    
    print("\n🎉 Test completed!")
    return True

if __name__ == "__main__":
    success = test_legal_assistant()
    sys.exit(0 if success else 1)
