import os
import streamlit as st
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain.memory import ConversationBufferMemory

from preprocessing import DocumentProcessor
from rag_pipeline import ConversationalRAG

## Suppress warnings seen during experimentation for a cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

def main():
    # Streamlit app setup/configuration
    st.set_page_config(page_title="Conversational PDF QA", layout="wide")
    st.title("📄 Conversational PDF QA with RAG")

    # Initialize session state if not already set
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    # LLM selection
    model_choice = st.selectbox("Choose an LLM:", options=["ollama", "groq", "gemini"], index=0)

    # Clear history button
    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

    # Initialize chat memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Restore memory from session state
    for speaker, message in st.session_state.chat_history:
        if speaker == "user":
            memory.chat_memory.add_user_message(message)
        elif speaker == "ai":
            memory.chat_memory.add_ai_message(message)

    # Only proceed if a file is uploaded
    if uploaded_file:
        # Avoid reprocessing the same file
        if uploaded_file.name != st.session_state.last_uploaded_file:
            st.session_state.last_uploaded_file = uploaded_file.name
            os.makedirs("documents", exist_ok=True)
            saved_path = os.path.join("documents", uploaded_file.name)

            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Processing document and building embeddings..."):
                processor = DocumentProcessor()
                vector_store = processor.process_pdf(file_path=saved_path)
                st.session_state.vector_store = vector_store # Save to avoid rebuilding again

            st.success("PDF processed and ready. Ask your question!")
        else:
            vector_store = st.session_state.vector_store
    else:
        st.info("Please upload a PDF file to begin.")
        st.stop()

    # Status messages
    log_messages = []
    status_box = st.empty()

    def update_status(msg):
        log_messages.append(msg)
        status_box.code("\n".join(log_messages), language="text")

    # Initialize the conversational RAG pipeline
    rag = ConversationalRAG(pdf_store=vector_store, 
                            model_choice=model_choice,
                            memory=memory, 
                            status_callback=update_status)

    # Show the latest Q&A chat between user and AI model
    if st.session_state.chat_history:
        user_msgs = [msg for (speaker, msg) in st.session_state.chat_history if speaker == "user"]
        ai_msgs = [msg for (speaker, msg) in st.session_state.chat_history if speaker == "ai"]
        if user_msgs:
            st.chat_message("user").write(user_msgs[-1])
        if ai_msgs:
            st.chat_message("assistant").write(ai_msgs[-1])

    # Input box
    query = st.chat_input("Ask a question about your document:")

    if query:
        log_messages.clear() # Clear previous logs

        with st.spinner("Thinking..."):
            response = rag.conversational_chat(query=query)

            # Display chat messages
            st.chat_message("user").write(query)
            st.chat_message("assistant").write(response)

            # Save to history
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("ai", response))

    # Expandable full history
    with st.expander("Show full chat history"):
        for speaker, message in st.session_state.chat_history:
            role = "🧑‍💻 You" if speaker == "user" else "⚙️ AI"
            st.markdown(f"**{role}:** {message}")

## Run main pipeline above
if __name__ == "__main__":
    main()

