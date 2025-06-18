import streamlit as st
import os
from dotenv import load_dotenv
from gre_vocab_agent import GREVocabAgent
from database import init_database

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="GRE Vocab Multi-Agent Tutor",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("🎓 GRE Vocab Multi-Agent Tutor")
    st.markdown("*Powered by LangGraph Multi-Agent Architecture*")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-openai-api-key-here":
        st.error("⚠️ Please set your OPENAI_API_KEY in the .env file to use this application.")
        st.info("Add your OpenAI API key to the .env file: `OPENAI_API_KEY=your-key-here`")
        return
    
    # Initialize database
    init_database()
    
    # Initialize the agent system
    if 'agent' not in st.session_state:
        st.session_state.agent = GREVocabAgent()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = """
        👋 Welcome to your GRE Vocab Tutor! I'm here to help you master GRE vocabulary through:
        
        🧠 **Quiz Mode**: Take adaptive quizzes that focus on your weak areas
        📖 **Study Mode**: Get personalized mnemonics and memory tricks
        📝 **Notes Mode**: Organize and update your vocabulary notes
        
        Just tell me what you'd like to do! For example:
        - "Start a quiz with 5 words"
        - "Help me study my weak words"
        - "Show me my notes"
        - "I want to take a quiz"
        """
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to do? (quiz, study, or notes)"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.process_message(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Sidebar with session info
    with st.sidebar:
        st.header("📊 Session Stats")
        
        if hasattr(st.session_state.agent, 'state'):
            state = st.session_state.agent.state
            quiz_perf = state.get('session_quiz_performance', {})
            
            if quiz_perf.get('words_attempted'):
                st.metric("Words Attempted", len(quiz_perf['words_attempted']))
                st.metric("Correct Answers", len(quiz_perf['correct_answers']))
                st.metric("Session Score", f"{quiz_perf.get('session_score', 0):.1%}")
                
                if quiz_perf.get('weak_words'):
                    st.subheader("🎯 Focus Words")
                    for word in quiz_perf['weak_words'][:5]:
                        st.write(f"• {word}")
        
        st.divider()
        
        if st.button("🔄 Reset Session"):
            st.session_state.agent = GREVocabAgent()
            st.session_state.messages = []
            st.rerun()
        
        if st.button("💾 Save Progress"):
            try:
                st.session_state.agent.save_session_progress()
                st.success("Progress saved!")
            except Exception as e:
                st.error(f"Error saving: {str(e)}")

if __name__ == "__main__":
    main()