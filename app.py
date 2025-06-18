import streamlit as st
import os
from gre_vocab_tutor import GREVocabAgent

def main():
    st.set_page_config(page_title="GRE Vocab Tutor", page_icon="📚")
    st.title("📚 GRE Vocab Tutor")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your-openai-api-key-here':
        st.error("❌ **OpenAI API Key Required**")
        st.write("Please set your OPENAI_API_KEY in the .env file")
        st.code("OPENAI_API_KEY=sk-your-actual-key-here")
        st.stop()
    
    # Test API connection
    if 'api_tested' not in st.session_state:
        with st.spinner("Testing OpenAI connection..."):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                # Test with a simple call
                client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                st.session_state.api_tested = True
                st.success("✓ OpenAI API connection verified")
            except Exception as e:
                st.error(f"❌ OpenAI API connection failed: {e}")
                st.write("Please check your API key and try again")
                st.stop()
    
    # Initialize agent
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = GREVocabAgent()
            st.success("✓ Agent initialized with LLM and RAG system")
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            st.stop()
    
    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to do?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get agent response
        try:
            response = st.session_state.agent.process_message(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()