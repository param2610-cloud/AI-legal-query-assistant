#!/usr/bin/env python3
"""
Simple CLI Interface for AI Legal Assistant
==========================================

A command-line interface for testing the legal assistant
without needing to run the full web application.
"""

import sys
import os
from pathlib import Path
import argparse

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import langchain
    except ImportError:
        missing_deps.append("langchain")
    
    try:
        import chromadb
    except ImportError:
        missing_deps.append("chromadb")
    
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n🔧 To install dependencies, run:")
        print("   pip install -r requirements.txt")
        print("\n📖 Or run the setup script:")
        print("   ./setup.sh")
        return False
    
    return True

def check_ollama():
    """Check if Ollama is available"""
    import subprocess
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, "Ollama command failed"
    except subprocess.TimeoutExpired:
        return False, "Ollama not responding"
    except FileNotFoundError:
        return False, "Ollama not installed"
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="AI Legal Assistant CLI")
    parser.add_argument("--check", action="store_true", help="Check system requirements")
    parser.add_argument("--setup", action="store_true", help="Show setup instructions")
    parser.add_argument("--demo", action="store_true", help="Run demo mode with sample questions")
    
    args = parser.parse_args()
    
    print("⚖️  AI Legal Assistant - CLI Interface")
    print("====================================")
    
    if args.check:
        print("\n🔍 Checking system requirements...")
        
        # Check Python dependencies
        print("\n📦 Python Dependencies:")
        if check_dependencies():
            print("   ✅ All Python dependencies available")
        else:
            return 1
        
        # Check Ollama
        print("\n🤖 Ollama LLM Service:")
        ollama_ok, ollama_info = check_ollama()
        if ollama_ok:
            print("   ✅ Ollama is available")
            print("   📋 Available models:")
            for line in ollama_info.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    print(f"      - {line.strip()}")
        else:
            print(f"   ❌ Ollama issue: {ollama_info}")
            print("   💡 Install Ollama: https://ollama.ai/download")
            return 1
        
        print("\n✅ System is ready!")
        return 0
    
    if args.setup:
        print("\n📋 Setup Instructions:")
        print("=====================")
        print("\n1. Install Ollama (Free Local LLM):")
        print("   curl -fsSL https://ollama.ai/install.sh | sh")
        print("\n2. Download a model:")
        print("   ollama pull llama3.2:3b")
        print("\n3. Install Python dependencies:")
        print("   pip install -r requirements.txt")
        print("\n4. Run the setup script:")
        print("   ./setup.sh")
        print("\n5. Start the application:")
        print("   python cli.py")
        print("   # OR")
        print("   streamlit run streamlit_app.py")
        return 0
    
    if args.demo:
        print("\n🎮 Demo Mode - Sample Legal Questions")
        print("===================================")
        
        sample_questions = [
            "What are my rights as a consumer if I receive a defective product?",
            "How do I register a complaint under the Consumer Protection Act?",
            "What are the child labor laws in India?",
            "How does the civil court procedure work?",
            "What documents are needed for marriage registration?"
        ]
        
        print("\n📝 Sample questions you can ask:")
        for i, question in enumerate(sample_questions, 1):
            print(f"{i}. {question}")
        
        print("\n💡 To get answers, initialize the full system and ask these questions!")
        return 0
    
    # Check system requirements first
    if not check_dependencies():
        print("\n💡 Run 'python cli.py --setup' for setup instructions")
        return 1
    
    ollama_ok, ollama_info = check_ollama()
    if not ollama_ok:
        print(f"\n❌ Ollama not available: {ollama_info}")
        print("💡 Run 'python cli.py --setup' for setup instructions")
        return 1
    
    # If all checks pass, try to run the assistant
    try:
        from agent.legal_assistant import SimpleLegalAssistant
        
        print("\n🚀 Initializing AI Legal Assistant...")
        assistant = SimpleLegalAssistant()
        
        if not assistant.initialize():
            print("❌ Failed to initialize the assistant")
            return 1
        
        print("✅ Assistant initialized successfully!")
        print("\n💬 Interactive Legal Assistant")
        print("=============================")
        print("Ask me any legal question or type 'quit' to exit.")
        print("Examples:")
        print("- 'What are my consumer rights?'")
        print("- 'How do I file a complaint for defective products?'")
        print("- 'What are the child labor protection laws?'")
        
        while True:
            print("\n" + "-" * 50)
            question = input("\n🗣️  Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\n👋 Thank you for using AI Legal Assistant!")
                break
            
            if not question:
                continue
            
            print("\n🤔 Analyzing your question...")
            result = assistant.ask_question(question)
            
            if result["status"] == "success":
                print(f"\n⚖️  Legal Assistant Response:")
                print("=" * 40)
                print(result["response"])
                
                # Show related acts
                print(f"\n📚 Related Legal Acts:")
                relevant_acts = assistant.get_relevant_acts(question)
                if relevant_acts:
                    for act in relevant_acts[:2]:  # Show top 2
                        print(f"• {act['act_name']}")
                else:
                    print("• No specific acts identified")
                    
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("💡 Run 'python cli.py --check' to verify installation")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
