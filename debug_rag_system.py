#!/usr/bin/env python3
"""
Debug script to check the LegalRAGSystem class and its methods
"""

import sys
from pathlib import Path

# Add the agent directory to path
agent_dir = Path(__file__).parent / "agent"
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

def debug_legal_rag_system():
    """Debug the LegalRAGSystem class"""
    try:
        print("Importing LegalRAGSystem...")
        from legal_assistant import LegalRAGSystem
        
        print("Creating LegalRAGSystem instance...")
        rag_system = LegalRAGSystem()
        
        print("Checking available methods...")
        methods = [method for method in dir(rag_system) if not method.startswith('_')]
        print(f"Available methods: {methods}")
        
        print("Checking if create_legal_qa_chain exists...")
        if hasattr(rag_system, 'create_legal_qa_chain'):
            print("✅ create_legal_qa_chain method exists!")
            print(f"Method type: {type(getattr(rag_system, 'create_legal_qa_chain'))}")
        else:
            print("❌ create_legal_qa_chain method NOT found!")
            
        print("Checking class structure...")
        print(f"Class: {rag_system.__class__}")
        print(f"MRO: {rag_system.__class__.__mro__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== LegalRAGSystem Debug Script ===")
    success = debug_legal_rag_system()
    if success:
        print("✅ Debug completed successfully")
    else:
        print("❌ Debug failed - check errors above")
