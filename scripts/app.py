import os
import hashlib
import streamlit as st
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain.memory import ConversationBufferMemory

from preprocessing import DocumentProcessor
from rag_pipeline import ConversationalRAG
from logger import get_logger

## Initialize a logger for logging purposes
logger =  get_logger()

## Suppress warnings seen during experimentation for a cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

## Helper function for initializing session state variables
def initialize_session_state():
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_uploaded_file", None)

## Helper function to generate an MD5 hash from the uploaded file content to detect changes
def get_file_hash(uploaded_file):
    """Generate an MD5 hash from the uploaded file content to detect changes."""
    return hashlib.md5(uploaded_file.getbuffer()).hexdigest()

## Helper function to initialize vector store
def load_or_process_pdf(uploaded_file, folder="documents"):
    os.makedirs(folder, exist_ok=True)
    saved_path = os.path.join(folder, uploaded_file.name)
    
    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    processor = DocumentProcessor()
    vector_store = processor.process_pdf(file_path=saved_path)
    logger.info(f"Processed and embedded file: {uploaded_file.name}")
    return vector_store

def main():
    # Streamlit app setup/configuration
    st.set_page_config(page_title="Conversational PDF QA", layout="wide")
    st.title("📄 Conversational PDF QA with RAG")

    # Initialize session state variabes if they don't already exist in the current
    # Streamlit session
    initialize_session_state()

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

    # Restore the previous chat into memory
    for speaker, message in st.session_state.chat_history:
        if speaker == "user":
            memory.chat_memory.add_user_message(message)
        elif speaker == "ai":
            memory.chat_memory.add_ai_message(message)

    # Status callbacks for Streamlit UI
    ui_log_messages = []
    status_box = st.empty()

    def log_status(msg, level: str = "info", print_to_ui: bool = False):
        """
        Logs a message to both the log file and the Streamlit UI status box.
        """
        # Skip printing a message to the Streamlit UI unless specified
        if print_to_ui:
            ui_log_messages.append(msg)
            status_box.code("\n".join(ui_log_messages), language="text")

        # Log to file
        if level == "debug":
            logger.debug(msg)
        elif level == "warning":
            logger.warning(msg)
        elif level == "error":
            logger.error(msg)
        else:
            logger.info(msg)        

    # Only proceed if a file is uploaded
    if uploaded_file:
        file_hash = get_file_hash(uploaded_file)

        # Avoid reprocessing the same file
        # if uploaded_file.name != st.session_state.last_uploaded_file:
        #     st.session_state.last_uploaded_file = uploaded_file.name

        # Avoid reprocessing the same file
        if file_hash != st.session_state.last_uploaded_file:
            st.session_state.last_uploaded_file = file_hash
            # os.makedirs("documents", exist_ok=True)
            # saved_path = os.path.join("documents", uploaded_file.name)

            try:
                with st.spinner("Processing document and building embeddings..."):
                    vector_store = load_or_process_pdf(uploaded_file)
                    st.session_state.vector_store = vector_store # Save to avoid rebuilding again
                st.success("PDF processed and ready. Ask your question!")

            except Exception as e:
                logger.exception(f"Error during document processing: {e}")
                st.error("An error occurred while processsing the PDF.")
                st.stop()

        # else:
        #     vector_store = st.session_state.vector_store

        else:
            try:
                vector_store = st.session_state.vector_store
            except KeyError:
                st.error("Vector store not found. Please re-upload the PDF.")
                return
            
    else:
        st.info("Please upload a PDF file to begin.")
        return

    try:
        # Initialize the conversational RAG pipeline
        rag = ConversationalRAG(pdf_store=vector_store, 
                                model_choice=model_choice,
                                memory=memory, 
                                status_callback=log_status)
    except Exception as e:
        logger.exception(f"Failed to initialize ConversationalRAG: {e}")
        st.error("An error occurred while initializing the chatbot.")
        return

    # Show the latest Q&A chat messages between user and AI model
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
        ui_log_messages.clear() # Clear previous logs
        log_status(f"User query received: {query}", print_to_ui=True)

        with st.spinner("Thinking..."):
            try:
                response = rag.conversational_chat(query=query)

                # Display chat messages
                st.chat_message("user").write(query)
                st.chat_message("assistant").write(response)

                # Save to history
                st.session_state.chat_history.append(("user", query))
                st.session_state.chat_history.append(("ai", response))

            except Exception as e:
                logger.exception(f"Error during chatbot response generation: {e}")
                st.error("An error occurred while the chatbot was generating a response.")

    # Expandable full history
    with st.expander("Show full chat history"):
        for speaker, message in st.session_state.chat_history:
            role = "🧑‍💻 You" if speaker == "user" else "⚙️ AI"
            st.markdown(f"**{role}:** {message}")

## Run main pipeline above
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Unhandled exception during app startup: {e}")
        st.error("An unexpected error occurred while starting the Streamlit app. Please check the logs.")

