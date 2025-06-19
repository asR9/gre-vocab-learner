import streamlit as st
import os
from gre_vocab_tutor import GREVocabAgent
from database import init_database, get_word_status_counts

def main():
    st.set_page_config(
        page_title="GRE Vocab Tutor", 
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize database
    init_database()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for API key (simplified)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your-openai-api-key-here':
        st.error("❌ **OpenAI API Key Required**")
        st.write("Please set your OPENAI_API_KEY in the .env file")
        st.stop()
    
    # Initialize agent (simplified)
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = GREVocabAgent()
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            st.stop()
    
    # Sidebar with dashboard
    with st.sidebar:
        st.markdown("### 📊 Quick Dashboard")
        
        try:
            status_counts = get_word_status_counts()
            total_words = sum(status_counts.values())
            
            if total_words > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Strong", status_counts['strong'])
                    st.metric("Weak", status_counts['weak'])
                with col2:
                    st.metric("Moderate", status_counts['moderate'])
                    st.metric("Unknown", status_counts['unknown'])
                
                mastery_pct = ((status_counts['strong'] + status_counts['moderate']) / total_words) * 100
                st.metric("Mastery", f"{mastery_pct:.1f}%")
            else:
                st.info("No data yet. Start with a quiz!")
        except:
            st.info("Dashboard loading...")
        
        st.markdown("---")
        st.markdown("### 🚀 Quick Actions")
        
        if st.button("📝 Quiz (5 words)", use_container_width=True):
            st.session_state.auto_message = "quiz 5"
            st.rerun()
        
        if st.button("📚 Study", use_container_width=True):
            st.session_state.auto_message = "study"
            st.rerun()
        
        if st.button("📋 View Notes", use_container_width=True):
            st.session_state.auto_message = "view notes"
            st.rerun()
        
        if st.button("📊 Progress", use_container_width=True):
            st.session_state.auto_message = "progress"
            st.rerun()
    
    # Main content
    st.markdown('<div class="main-header">📚 GRE Vocab Tutor</div>', unsafe_allow_html=True)
    
    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Handle auto messages from sidebar
    if 'auto_message' in st.session_state:
        auto_msg = st.session_state.auto_message
        del st.session_state.auto_message
        
        # Add to messages and process
        st.session_state.messages.append({"role": "user", "content": auto_msg})
        response = st.session_state.agent.process_message(auto_msg)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to do? (quiz, study, notes, progress)"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.process_message(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)

if __name__ == "__main__":
    main()