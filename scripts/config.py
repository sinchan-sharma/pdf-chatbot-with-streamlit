import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from logger import get_logger

## Initialize a logger for logging purposes
logger =  get_logger()

## Load environment variables depending on local run or inside Docker
## Also controls where logging occurs
if os.getenv("RUNNING_IN_DOCKER") == "true":
    logger.info("Running inside Docker - logging to console only.")
else:
    load_dotenv()
    logger.info("Running locally - logging to both file and console.")

## Get API keys and other environment variables from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")

## Check whether user has necessary API keys set up in their environment
if not GOOGLE_API_KEY or not GROQ_API_KEY:
    logger.error("Missing API keys for Google and/or Groq.")
    raise ValueError("Please ensure both GOOGLE_API_KEY and GROQ_API_KEY are set up in your .env file.")

## Check that LangSmith API key is properly set up
if not LANGSMITH_API_KEY:
    logger.warning("Warning: LangSmith API key is not set. LangSmith features will not work.")

## Check for LangSmith tracing
if LANGSMITH_API_KEY and not LANGSMITH_TRACING:
    logger.warning("Warning: LangSmith tracing is not enabled. This may affect tracing-related features.")

## Constants
DOCUMENTS_FOLDER = os.path.join(".", "documents")
VECTOR_DB_DIR = os.path.join(".", "chroma_db")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
COLLECTION_PDF = "pdf-collection"
COLLECTION_NONPDF = "nonpdf-collection"

## Embedding model used
HF_EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## Instructions prompt for dynamic prompting. This is used to dictate how the LLM answers 
## user questions using different guidelines for different scenarios
INSTRUCTIONS = {
    "Factual": """
You are a helpful AI assistant tasked with answering factual questions using the provided context.

    **Answering Guidelines:**

    1. **If the answer is clearly available or can be paraphrased from the context**:
    - Extract the relevant information and answer directly based on it, even if the wording is not exact.

    2. **If the context does not include the answer at all**:
    - For **targeted questions** (e.g., “What were the findings of this paper?”):
        - Respond: _"The provided context does not include that information."_
        - If there are subtle clues, offer a speculative answer starting with:  
        _"Note: This is a speculative answer based on limited context..."_

    - For **general questions** (e.g., "What is AI?" or "What is artificial intelligence?"):
        - If the context lacks a full answer, you may respond using your general knowledge **only after confirming** the context is not sufficient.
        - In that case, begin with:  
        _"Note: The provided context did not contain this directly, so this is based on general knowledge."_

    **Important:**
    - Make a strong effort to extract relevant information from the context first.
    - Do **not** rely on internal knowledge unless it is truly not in the provided context.
    - Always indicate the source: context, inferred, or general knowledge.
    - Provide complete and well-formed answers in full sentences.
    - If offering a speculative answer for a targeted question, re-emphasize at the end that the answer should be taken with a grain of salt. 
    """,

    "Interpretive": """
    You are a helpful AI assistant tasked with answering interpretive questions using the provided context.

    **Answering Guidelines:**

    1. **Strongly prioritize the provided context** when forming your interpretation.
    - Use the context to synthesize, summarize, and explain broader themes or implications.
    - Make every effort to ground your response in the information given.

    2. **If the context is incomplete or only partially relevant**:
    - Use the available context to form a thoughtful answer.
    - Clearly state when your answer is based on limited context, e.g.:
        _"Based on the information available in the provided context..."_
    - Re-emphasize at the end that the answer should be taken with a grain of salt.

    3. **Only if the context does not contain enough relevant information** to address the question:
    - For targeted questions (e.g., about specific conclusions or points):
        - Respond clearly: _"The provided context does not contain sufficient information to answer this question."_
    - For broader or more general questions:
        - You may answer based on your internal knowledge, but **include a disclaimer**:
        _"Note: This answer is based on general knowledge, as the provided context does not address this directly."_

    **Important:**
    - Do not invent or hallucinate information unsupported by context.
    - Always clarify whether your answer is grounded in context, limited context, or general knowledge.
    - Provide clear, coherent, and well-structured answers.
    """
}

## Classification prompt template used for the LLM to detect question type 
## using few-shot examples.
QUESTION_CLASSIFIER_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""
You are a classifier that determines whether a user question is Factual or Interpretive.

- A *Factual* question asks for specific information found directly in a document (e.g., names, dates, techniques, facts).
- An *Interpretive* question asks for broader meaning, implications, or synthesis (e.g., summaries, themes, significance).

Respond with only the single word: Factual or Interpretive.

Examples:

Question: What is artificial intelligence?
Classification: Factual

Question: What is this paper mainly about?
Classification: Interpretive

Question: What are some of the key findings mentioned in this study?
Classification: Factual

Question: How does this paper relate to broader trends in artificial intelligence?
Classification: Interpretive

Now classify the following question:
Question: {query}
Classification:
"""
)

