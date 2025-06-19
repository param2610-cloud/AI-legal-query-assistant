"""
Streamlit Web Interface for AI Legal Assistant
============================================

A user-friendly web interface for the legal assistant that helps
common people understand Indian laws in simple terms.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from agent.legal_assistant import SimpleLegalAssistant
    SYSTEM_AVAILABLE = True
except ImportError as e:
    SYSTEM_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Page configuration
st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E4057;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
        background-color: #f8f9fa;
    }
    .legal-tip {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">⚖️ AI Legal Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Understanding Indian Laws Made Simple</p>', unsafe_allow_html=True)
    
    # Check if system is available
    if not SYSTEM_AVAILABLE:
        st.error(f"""
        ❌ **System Not Ready**
        
        The AI Legal Assistant requires additional setup. Please follow these steps:
        
        1. **Install Ollama** (Free Local LLM):
           ```bash
           # Install Ollama
           curl -fsSL https://ollama.ai/install.sh | sh
           
           # Pull a model (choose one)
           ollama pull llama3.2:3b      # Recommended for good performance
           ollama pull mistral:7b       # Alternative option
           ollama pull codellama:7b     # For code-heavy legal docs
           ```
        
        2. **Install Python Dependencies**:
           ```bash
           pip install -r requirements.txt
           ```
        
        3. **Restart this application**
        
        **Error Details:** {IMPORT_ERROR}
        """)
        
        st.info("""
        💡 **Why These Tools?**
        - **Ollama**: Runs LLMs locally for free (no API costs)
        - **ChromaDB**: Free vector database for document search
        - **Sentence Transformers**: Free embedding models
        - **LangChain**: Framework for building AI applications
        """)
        return
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
        st.session_state.initialized = False
        st.session_state.chat_history = []
    
    # Sidebar for system status and controls
    with st.sidebar:
        st.header("🔧 System Status")
        
        if not st.session_state.initialized:
            st.warning("⏳ System not initialized")
            if st.button("🚀 Initialize System", type="primary"):
                with st.spinner("Initializing AI Legal Assistant..."):
                    st.session_state.assistant = SimpleLegalAssistant()
                    success = st.session_state.assistant.initialize()
                    
                    if success:
                        st.session_state.initialized = True
                        st.success("✅ System initialized successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to initialize system")
        else:
            st.success("✅ System ready")
            if st.button("🔄 Restart System"):
                st.session_state.assistant = None
                st.session_state.initialized = False
                st.session_state.chat_history = []
                st.rerun()
        
        # Legal areas covered
        st.header("📚 Legal Areas Covered")
        legal_areas = [
            "Consumer Protection",
            "Child Labor Laws",
            "Civil Procedures",
            "Marriage & Family Laws",
            "Drug & Cosmetics Act",
            "Dowry Prohibition",
            "Birth, Death & Marriage Registration"
        ]
        
        for area in legal_areas:
            st.markdown(f"• {area}")
        
        # Disclaimer
        st.header("⚠️ Important Disclaimer")
        st.markdown("""
        <div class="warning-box">
        <strong>This is an AI assistant for educational purposes only.</strong>
        
        • Not a substitute for professional legal advice
        • Consult a qualified lawyer for complex matters
        • Laws may change - verify current regulations
        • Use at your own discretion
        </div>
        """, unsafe_allow_html=True)
    
    # Main interface
    if not st.session_state.initialized:
        st.info("👈 Please initialize the system using the sidebar to get started.")
        
        # Show example scenarios while waiting
        st.header("🎯 What You Can Ask")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Consumer Rights:**
            - "What are my rights when buying products online?"
            - "My shop sold me a defective item, what can I do?"
            - "How do I file a consumer complaint?"
            
            **Employment Issues:**
            - "My employer is not paying minimum wage"
            - "What are the child labor laws in India?"
            - "Can I be fired without notice?"
            """)
        
        with col2:
            st.markdown("""
            **Family Matters:**
            - "How do I register a marriage?"
            - "What are the dowry prohibition laws?"
            - "How to register birth/death certificates?"
            
            **Legal Procedures:**
            - "How does civil court procedure work?"
            - "What documents do I need for filing a case?"
            - "How long do legal proceedings take?"
            """)
        
        return
    
    # Chat interface
    st.header("💬 Ask Your Legal Question")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("📋 Conversation History")
        for chat in st.session_state.chat_history:
            with st.container():
                st.markdown(f"**You:** {chat['question']}")
                st.markdown(f"**Legal Assistant:** {chat['response']}")
                st.markdown("---")
    
    # Input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_question = st.text_area(
            "Describe your legal question or situation:",
            placeholder="For example: 'I bought a phone online but it's defective. The seller is refusing to refund. What are my rights?'",
            height=100
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        ask_button = st.button("🔍 Ask Question", type="primary")
        clear_button = st.button("🗑️ Clear Chat")
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if ask_button and user_question.strip():
        with st.spinner("🤔 Analyzing your question..."):
            result = st.session_state.assistant.ask_question(user_question)
            
            if result["status"] == "success":
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "response": result["response"]
                })
                
                # Display the response
                st.markdown('<div class="chat-message">', unsafe_allow_html=True)
                st.markdown(f"**Your Question:** {user_question}")
                st.markdown(f"**Legal Assistant Response:**")
                st.markdown(result["response"])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show relevant acts
                st.subheader("📖 Related Legal Acts")
                relevant_acts = st.session_state.assistant.get_relevant_acts(user_question)
                
                if relevant_acts:
                    for act in relevant_acts[:3]:  # Show top 3 most relevant
                        with st.expander(f"📜 {act['act_name']}"):
                            for section in act['relevant_sections'][:2]:  # Show top 2 sections
                                st.markdown(f"**Section {section['section_id']}:** {section['title']}")
                                st.markdown(section['content_preview'])
                                st.markdown("---")
                
            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error occurred')}")
    
    # Quick actions
    st.header("🚀 Quick Actions")
    
    quick_questions = [
        "What are my consumer rights?",
        "How do I file a complaint against unfair trade practices?",
        "What are the child labor laws in India?",
        "How does civil court procedure work?",
        "What are the marriage registration requirements?"
    ]
    
    cols = st.columns(len(quick_questions))
    for i, question in enumerate(quick_questions):
        with cols[i]:
            if st.button(question, key=f"quick_{i}"):
                with st.spinner("Processing..."):
                    result = st.session_state.assistant.ask_question(question)
                    if result["status"] == "success":
                        st.session_state.chat_history.append({
                            "question": question,
                            "response": result["response"]
                        })
                        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🏛️ AI Legal Assistant - Making Indian Laws Accessible to Everyone</p>
        <p>Built with ❤️ using Open Source Tools</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
