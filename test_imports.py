#!/usr/bin/env python3
"""
Test script to verify all imports are working correctly
"""

import sys
from pathlib import Path

print("🧪 Testing Import Issues")
print("=" * 50)

# Test 1: Test Indian Kanoon Client Import
print("1. Testing Indian Kanoon Client Import...")
try:
    from indian_kanoon_client import IndianKanoonClient, create_indian_kanoon_client
    print("   ✅ Indian Kanoon Client imported successfully")
except ImportError as e:
    print(f"   ❌ Indian Kanoon Client import failed: {e}")

# Test 2: Test Legal Assistant Import
print("\n2. Testing Legal Assistant Import...")
try:
    from agent.legal_assistant import SimpleLegalAssistant
    print("   ✅ Legal Assistant imported successfully")
except ImportError as e:
    print(f"   ❌ Legal Assistant import failed: {e}")

# Test 3: Test Training Data Loading
print("\n3. Testing Training Data Loading...")
try:
    from agent.legal_assistant import LegalDataProcessor
    
    processor = LegalDataProcessor()
    print("   ✅ LegalDataProcessor created successfully")
    
    # Test loading training data
    training_docs = processor.load_training_data()
    print(f"   ✅ Training data loaded: {len(training_docs)} documents")
    
except Exception as e:
    print(f"   ❌ Training data loading failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test Basic Legal Assistant Initialization
print("\n4. Testing Legal Assistant Initialization...")
try:
    from agent.legal_assistant import SimpleLegalAssistant
    
    assistant = SimpleLegalAssistant()
    print("   ✅ Legal Assistant created successfully")
    
    # Note: We won't call initialize() as it takes time and resources
    print("   ℹ️  Full initialization skipped (takes time)")
    
except Exception as e:
    print(f"   ❌ Legal Assistant creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("🎉 Import testing completed!")
