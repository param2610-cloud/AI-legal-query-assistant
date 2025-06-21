"""
Enhanced Streamlit Web Interface for AI Legal Assistant with Case Law Analysis
============================================================================

An advanced web interface that provides:
1. Basic legal assistance from Indian acts
2. Case law precedent analysis
3. Strategic legal guidance
4. Actionable recommendations with next steps
"""

import streamlit as st
import sys
import os
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the parent directory to the path to import our modules
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "agent"))

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.assistant = None
    st.session_state.case_law_available = False
    st.session_state.conversation_history = []

@st.cache_resource
def initialize_system():
    """Initialize the legal assistant system"""
    try:
        # Try to import enhanced system first
        from enhanced_case_law_analyzer import CaseLawAnalyzer
        from legal_assistant import SimpleLegalAssistant, LegalDataProcessor, LegalRAGSystem
        
        # Initialize case law analyzer
        case_analyzer = CaseLawAnalyzer()
        
        # Initialize basic legal assistant
        assistant = SimpleLegalAssistant()
        
        if assistant.initialize():
            return {
                'assistant': assistant,
                'case_analyzer': case_analyzer,
                'case_law_available': True,
                'status': 'Enhanced system initialized successfully'
            }
        else:
            return {
                'assistant': None,
                'case_analyzer': None,
                'case_law_available': False,
                'status': 'Failed to initialize enhanced system'
            }
            
    except ImportError as e:
        # Fallback to basic system
        try:
            from agent.legal_assistant import SimpleLegalAssistant
            
            assistant = SimpleLegalAssistant()
            if assistant.initialize():
                return {
                    'assistant': assistant,
                    'case_analyzer': None,
                    'case_law_available': False,
                    'status': 'Basic system initialized (case law analysis not available)'
                }
            else:
                return {
                    'assistant': None,
                    'case_analyzer': None,
                    'case_law_available': False,
                    'status': 'Failed to initialize system'
                }
        except ImportError as e2:
            return {
                'assistant': None,
                'case_analyzer': None,
                'case_law_available': False,
                'status': f'System not available: {str(e2)}'
            }

# Page configuration
st.set_page_config(
    page_title="AI Legal Assistant - Enhanced",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .case-law-section {
        background-color: #e3f2fd;
        border: 1px solid #90caf9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .strategic-guidance {
        background-color: #fff3e0;
        border: 1px solid #ffcc02;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .precedent-card {
        background-color: #f3e5f5;
        border: 1px solid #ce93d8;
        border-radius: 0.5rem;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    .action-item {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

def display_header():
    """Display the application header"""
    st.markdown('<h1 class="main-header">⚖️ AI Legal Assistant - Enhanced with Case Law Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p><strong>Get comprehensive legal guidance with case law precedents and strategic recommendations</strong></p>
        <p style="color: #666;">🆓 Free • 🤖 AI-Powered • 📚 Case Law Analysis • 🎯 Strategic Guidance</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar with system information and options"""
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Initialize system if not done
        if not st.session_state.initialized:
            with st.spinner("Initializing AI Legal Assistant..."):
                system_info = initialize_system()
                st.session_state.assistant = system_info['assistant']
                st.session_state.case_analyzer = system_info.get('case_analyzer')
                st.session_state.case_law_available = system_info['case_law_available']
                st.session_state.initialized = True
                
                if system_info['status']:
                    if st.session_state.case_law_available:
                        st.success("✅ Enhanced system ready!")
                        st.info("📚 Case law analysis available")
                    else:
                        st.warning("⚠️ Basic system ready")
                        st.info("Case law analysis not available")
                else:
                    st.error("❌ System initialization failed")
        
        # Display system capabilities
        st.subheader("🎯 Available Features")
        
        if st.session_state.assistant:
            st.write("✅ Legal document search")
            st.write("✅ Plain language explanations")
            st.write("✅ Contextual guidance")
            
            if st.session_state.case_law_available:
                st.write("✅ Case law precedent analysis")
                st.write("✅ Strategic legal guidance")
                st.write("✅ Actionable recommendations")
                st.write("✅ Risk assessment")
            else:
                st.write("❌ Case law analysis (not available)")
                st.write("❌ Strategic guidance (not available)")
        else:
            st.write("❌ System not available")
        
        # Analysis options
        st.subheader("🔍 Analysis Options")
        
        include_precedents = st.checkbox(
            "Include case law precedents", 
            value=st.session_state.case_law_available,
            disabled=not st.session_state.case_law_available,
            help="Search for similar legal precedents"
        )
        
        include_strategy = st.checkbox(
            "Generate strategic guidance", 
            value=st.session_state.case_law_available,
            disabled=not st.session_state.case_law_available,
            help="Provide actionable legal strategy"
        )
        
        include_next_steps = st.checkbox(
            "Show next steps", 
            value=True,
            help="Display recommended actions"
        )
        
        # Legal areas
        st.subheader("📚 Covered Legal Areas")
        legal_areas = [
            "Consumer Protection",
            "Employment Rights",
            "Family & Marriage Laws",
            "Property Rights",
            "Child Protection Laws",
            "Civil Procedures",
            "Criminal Law Basics",
            "Constitutional Rights"
        ]
        
        for area in legal_areas:
            st.write(f"• {area}")
        
        # Disclaimer
        st.subheader("⚠️ Important Notice")
        st.warning("""
        This AI assistant provides general legal information for educational purposes only. 
        
        **Always consult a qualified lawyer for:**
        - Official legal advice
        - Court proceedings
        - Complex legal matters
        - Document preparation
        """)
        
        return {
            'include_precedents': include_precedents,
            'include_strategy': include_strategy,
            'include_next_steps': include_next_steps
        }

async def get_enhanced_response(query: str, options: Dict[str, bool]) -> Dict[str, Any]:
    """Get enhanced response with case law analysis"""
    
    result = {
        'basic_response': '',
        'precedents': [],
        'strategy': None,
        'next_steps': [],
        'error': None
    }
    
    try:
        # Get basic response
        if st.session_state.assistant:
            basic_response = st.session_state.assistant.ask_question(query)
            result['basic_response'] = basic_response.get('response', 'No response generated')
        
        # Get case law analysis if available and requested
        if st.session_state.case_analyzer and options.get('include_precedents'):
            try:
                # Find similar precedents
                similar_cases = st.session_state.case_analyzer._find_similar_cases(query, top_k=3)
                result['precedents'] = similar_cases
                
                # Get strategic guidance if requested
                if options.get('include_strategy'):
                    strategy = await st.session_state.case_analyzer.analyze_legal_situation(query)
                    result['strategy'] = strategy
                    
            except Exception as e:
                result['error'] = f"Case law analysis error: {str(e)}"
        
        return result
        
    except Exception as e:
        result['error'] = f"System error: {str(e)}"
        return result

def display_basic_response(response: str):
    """Display the basic legal response"""
    st.markdown("### 📋 Legal Analysis")
    st.markdown(f'<div class="chat-message">{response}</div>', unsafe_allow_html=True)

def display_precedents(precedents: List[Any]):
    """Display case law precedents"""
    if not precedents:
        return
    
    st.markdown("### 📚 Relevant Case Law Precedents")
    
    for i, case in enumerate(precedents, 1):
        with st.expander(f"📖 Case {i}: {case.case_name} ({case.year})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Court:** {case.court}")
                st.write(f"**Citation:** {case.citation}")
                st.write(f"**Legal Principle:** {case.legal_principle}")
                
                if case.facts:
                    st.write(f"**Facts:** {case.facts[:300]}...")
                
                if case.keywords:
                    st.write(f"**Keywords:** {', '.join(case.keywords[:5])}")
            
            with col2:
                relevance_pct = int(case.relevance_score * 100) if hasattr(case, 'relevance_score') else 0
                st.metric("Relevance", f"{relevance_pct}%")
                
                st.write(f"**Category:** {case.category.value if hasattr(case, 'category') else 'N/A'}")

def display_strategic_guidance(strategy):
    """Display strategic legal guidance"""
    if not strategy:
        return
    
    st.markdown("### 🎯 Strategic Legal Guidance")
    
    # Immediate Actions
    if strategy.immediate_actions:
        st.markdown("#### ⚡ Immediate Actions Required")
        for action in strategy.immediate_actions:
            st.markdown(f'<div class="action-item">• {action}</div>', unsafe_allow_html=True)
    
    # Legal Remedies
    if strategy.legal_remedies:
        st.markdown("#### ⚖️ Available Legal Remedies")
        for remedy in strategy.legal_remedies:
            st.write(f"• {remedy}")
    
    # Required Documentation
    if strategy.required_documentation:
        st.markdown("#### 📄 Required Documentation")
        for doc in strategy.required_documentation:
            st.write(f"• {doc}")
    
    # Timeline and Costs
    col1, col2 = st.columns(2)
    
    with col1:
        if strategy.timeline:
            st.markdown("#### ⏰ Expected Timeline")
            for phase, time in strategy.timeline.items():
                st.write(f"**{phase}:** {time}")
    
    with col2:
        if strategy.estimated_costs:
            st.markdown("#### 💰 Estimated Costs")
            for item, cost in strategy.estimated_costs.items():
                st.write(f"**{item}:** {cost}")
    
    # Risks and Success Probability
    if strategy.risks_and_challenges or strategy.success_probability:
        st.markdown("#### ⚠️ Risk Assessment")
        
        if strategy.success_probability:
            st.info(f"**Success Probability:** {strategy.success_probability}")
        
        if strategy.risks_and_challenges:
            st.markdown("**Key Risks:**")
            for risk in strategy.risks_and_challenges:
                st.write(f"• {risk}")
    
    # Alternative Options
    if strategy.alternative_options:
        st.markdown("#### 🔄 Alternative Options")
        for option in strategy.alternative_options:
            st.write(f"• {option}")

def display_conversation_history():
    """Display conversation history"""
    if st.session_state.conversation_history:
        st.markdown("### 💬 Conversation History")
        
        for i, entry in enumerate(reversed(st.session_state.conversation_history[-5:]), 1):
            with st.expander(f"Question {len(st.session_state.conversation_history) - i + 1}: {entry['query'][:50]}..."):
                st.write(f"**Q:** {entry['query']}")
                st.write(f"**A:** {entry['response'][:300]}...")
                st.write(f"**Time:** {entry['timestamp']}")

def main():
    """Main application function"""
    
    # Display header
    display_header()
    
    # Display sidebar and get options
    options = display_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Ask Your Legal Question")
        
        # Sample questions
        sample_questions = [
            "My employer is not paying overtime despite 12-hour shifts. What are my rights?",
            "I bought a defective product online and seller refuses refund. What can I do?",
            "My husband is demanding dowry after marriage. How can I protect myself?",
            "My child's birth registration is delayed by 2 years. What's the process now?",
            "Can I file a consumer complaint for poor service quality?"
        ]
        
        selected_sample = st.selectbox(
            "💡 Try a sample question:",
            [""] + sample_questions,
            key="sample_selector"
        )
        
        # Query input
        user_query = st.text_area(
            "Enter your legal question:",
            value=selected_sample if selected_sample else "",
            height=100,
            placeholder="Describe your legal situation in detail..."
        )
        
        # Submit button
        if st.button("🔍 Get Legal Guidance", type="primary"):
            if not user_query.strip():
                st.warning("Please enter your legal question.")
                return
            
            if not st.session_state.assistant:
                st.error("Legal assistant system is not available. Please check the system setup.")
                return
            
            # Show processing
            with st.spinner("Analyzing your legal question..."):
                
                # Get response
                if st.session_state.case_law_available:
                    # Use async for enhanced analysis
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(get_enhanced_response(user_query, options))
                    finally:
                        loop.close()
                else:
                    # Basic response only
                    result = {
                        'basic_response': st.session_state.assistant.ask_question(user_query).get('response', ''),
                        'precedents': [],
                        'strategy': None,
                        'error': None
                    }
                
                # Display results
                if result.get('error'):
                    st.error(f"Error: {result['error']}")
                
                # Basic response
                if result.get('basic_response'):
                    display_basic_response(result['basic_response'])
                
                # Case law precedents
                if result.get('precedents') and options.get('include_precedents'):
                    display_precedents(result['precedents'])
                
                # Strategic guidance
                if result.get('strategy') and options.get('include_strategy'):
                    display_strategic_guidance(result['strategy'])
                
                # Save to conversation history
                from datetime import datetime
                st.session_state.conversation_history.append({
                    'query': user_query,
                    'response': result.get('basic_response', ''),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'has_precedents': len(result.get('precedents', [])) > 0,
                    'has_strategy': result.get('strategy') is not None
                })
    
    with col2:
        # Display conversation history
        display_conversation_history()
        
        # Quick tips
        st.markdown("### 💡 Quick Tips")
        st.info("""
        **For better results:**
        • Be specific about your situation
        • Mention relevant dates and amounts
        • Include location if relevant
        • Describe what you've tried already
        """)
        
        # Emergency contacts
        st.markdown("### 🆘 Emergency Legal Help")
        st.warning("""
        **For urgent legal matters:**
        • Legal Aid: 15100
        • Women Helpline: 181
        • Child Helpline: 1098
        • Consumer Helpline: 1915
        """)

if __name__ == "__main__":
    main()
