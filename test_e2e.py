#!/usr/bin/env python3
"""
Test script to verify the legal assistant works end-to-end
"""

import sys
from pathlib import Path

print("🧪 Testing Legal Assistant End-to-End")
print("=" * 60)

try:
    from agent.legal_assistant import SimpleLegalAssistant
    
    print("1. Creating Legal Assistant...")
    assistant = SimpleLegalAssistant()
    
    print("2. Initializing system (this will take a few minutes)...")
    if assistant.initialize():
        print("   ✅ System initialized successfully!")
        
        print("\n3. Testing a simple legal question...")
        test_question = "What are my consumer rights?"
        
        result = assistant.ask_question(test_question)
        
        if result and result.get('status') == 'success':
            print("   ✅ Question processed successfully!")
            print(f"   📝 Question: {result['question']}")
            print(f"   💬 Response: {result['response'][:200]}...")
            print(f"   🔍 Legal terms found: {result['terms_count']}")
        else:
            print(f"   ❌ Question processing failed: {result}")
    else:
        print("   ❌ System initialization failed")

except KeyboardInterrupt:
    print("\n⏹️  Test interrupted by user")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🎉 End-to-end testing completed!")
