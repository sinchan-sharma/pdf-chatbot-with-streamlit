from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableSequence, Runnable, RunnableLambda
# from langchain_core.runnables import RunnableMap
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM

from config import INSTRUCTIONS, QUESTION_CLASSIFIER_PROMPT, GROQ_API_KEY, GOOGLE_API_KEY
import traceback

class ConversationalRAG:
    """
    A Conversational RAG pipeline that retrieves documents from a single vector store,
    formats them with instructions, and passes the result to either an Ollama, Gemini,
    or Groq LLM for answering user queries.
    """
    def __init__(self, pdf_store, model_choice="ollama", memory=None, status_callback=None):
        """
        Initialize the ConversationalRAG pipeline
        """
        self.vector_store_pdf = pdf_store
        self.model_choice = model_choice
        self.llm = self._get_llm(model_choice)
        self.memory = memory or ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.template = ChatPromptTemplate.from_messages([
            ("system", "{instruction}\n\nYou are a helpful AI assistant..."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        self.status_callback = status_callback or (lambda msg: None) # fallback to no-op

    def log(self, msg: str):
        """Log message to both callback and stdout (optional)."""
        self.status_callback(msg)
        print(msg) # This is optional of you want a cleaner terminal

    # LLM selection
    def _get_llm(self, name): 
        """
        Load the LLM based on the model name (either 'ollama' or 'groq').
        """
        if name == "ollama":
            return OllamaLLM(model="gemma3", temperature=0.3)
        if name == "groq":
            return ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, api_key=GROQ_API_KEY)
        if name == "gemini":
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=GOOGLE_API_KEY)
        raise ValueError(f"Unsupported model choice: '{name}'. Use 'Ollama' 'Gemini', or 'Groq'.")

    # Question classification as 'Factual' or 'Interpretive'
    def _classify(self, query):
        """
        Uses the LLM to classify a question/query as 'Factual' or 'Interpretive'
        """
        try: 
            return (QUESTION_CLASSIFIER_PROMPT | self.llm | StrOutputParser()).invoke({"query": query}).strip()
        except Exception as e: 
            self.log(f"An error occurred during question classification: {e}")
            return "Interpretive"

    # Format retrieved documents/chunks as a single string to pass to the LLM
    def _format_docs(self, docs):
        """
        Join retrieved documents/chunks as a single string of context to pass
        to the LLM.
        """
        if not docs:
            self.log("No documents to format.")
            return "Warning: No relevant context was found."
        
        # Check document types and contents for debugging (optional)
        for i, doc in enumerate(docs):
            if not hasattr(doc, "page_content"):
                self.log(f"Document at index {i} missing 'page_content'. Actual type: {type(doc)}")
            elif not isinstance(doc.page_content, str):
                self.log(f"Document page_content at index {i} is not a string. Type: {type(doc.page_content)}")

        try:
            formatted = "\n\n".join(doc.page_content for doc in docs if hasattr(doc, "page_content"))
            return formatted
        except Exception as e:
            self.log(f"[ERROR] Exception in _format_docs: {e}")
            traceback.print_exc()
            return "Warning: Error formatting document contents."

    # Build the RAG chain
    def _build_chain(self, retriever, query, question_type: str):
        """
        Build the RAG chain with memory, instructions, retrieved context, and
        the chosen LLM.
        """
        # Have the LLM infer the question/query type
        question_type = self._classify(query)
        # Get instruction text for this question type, defaulting to "Interpretive"
        instruction = INSTRUCTIONS.get(question_type, INSTRUCTIONS["Interpretive"])
        # Load chat history from memory
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])

        self.log(f"Classified query as: {question_type}")
        self.log(f"Building chain with question_type: {question_type}")
        self.log(f"Instruction text length: {len(instruction)}")
        self.log(f"Number of chat history messages: {len(chat_history)}")

        # Retrieve relevant documents from the retriever
        docs = retriever.get_relevant_documents(query)
        self.log(f"Retrieved {len(docs)} docs.")
        
        context_str = self._format_docs(docs)
        self.log(f"Context string length: {len(context_str)}")

        # Prompt inputs prepared separately for clarity
        prompt_inputs = {"instruction": instruction, "context": context_str,
                         "question": query, "chat_history": chat_history}

        # Build chain using a RunnableMap over the entire prompt_inputs dictionary
        chain = (
            RunnableLambda(lambda _: prompt_inputs) 
            | self.template 
            | self.llm
            | StrOutputParser()
            )
        
        return RunnableWithMessageHistory(
                    RunnableSequence(chain), 
                    get_session_history=lambda _: self.memory.chat_memory,
                    input_messages_key="question",
                    history_messages_key="chat_history"
                    )

    # Run the full pipeline
    def conversational_chat(self, query: str, k: int = 5):
        """
        Run the full conversational RAG pipeline and return the model's response
        """
        # print(f"conversational_chat called with query='{query}' and k={k}")
        self.log(f"Received query: '{query}'")

        # Have the LLM infer the question/query type
        question_type = self._classify(query)
        retriever = self.vector_store_pdf.as_retriever(search_kwargs={"k":k})
        chain = self._build_chain(retriever, query, question_type)

        try:
            self.log("Invoking LLM chain...")
            # Invoke the LLM chain and try to get a response
            response = chain.invoke({"question": query}, 
                                     config={"configurable": {"session_id": "default"}})
            self.log("LLM response received successfully.")
        except Exception as e:
            print(f"[ERROR] Exception during chain.invoke: {e}")
            traceback.print_exc()
            response = f"An error occurred while processing your query: {e}"

        return response

