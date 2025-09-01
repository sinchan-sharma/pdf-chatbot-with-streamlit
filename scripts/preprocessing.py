import os
import shutil
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

from config import VECTOR_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, HF_EMBEDDINGS
from logger import get_logger

## Initialize a logger for logging purposes
logger = get_logger()

## File that stores the hash of the last processed PDF file
METADATA_FILE = os.path.join(VECTOR_DB_DIR, "last_file.txt")

## Clear the contents of the vector store before rebuilding
## This is only necessary if a new PDF file is uploaded during the same session
def clear_vector_store(path):
    """
    Clear the vector store directory. This ensures that there's no leftover data
    from a previous document that could pollute the new vector store.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        logger.info(f"Cleared vector store at: {path}")

## Helper to generate a hash for the uploaded file's contents
## This is to avoid rebuilding the vector store if the uploaded PDF file hasn't changed
def get_file_hash(file_path):
    """
    Generate a hash of the file contents to detect if the file has changed.
    This helps to avoid rebuilding the vector store if the uploaded PDF file hasn't changed.
    """
    try:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            content = f.read()
            hasher.update(content)
        return hasher.hexdigest()
    except Exception as e:
        logger.exception(f"Error generating hash for file {file_path}: {e}")
        raise


## Create a DocumentProcessor class for processing PDF documents
class DocumentProcessor:
    """
    Load a PDF, chunk it using a text splitter, and embed it into a Chroma vector store
    using HF embeddings.
    """

    def __init__(self, persist_dir=VECTOR_DB_DIR, chunk_size=CHUNK_SIZE,
                 chunk_overlap=CHUNK_OVERLAP, hf_embeddings=HF_EMBEDDINGS):

        self.persist_dir = persist_dir      # Root directory for vector store persistence
        self.chunk_size = chunk_size        # Chunk size
        self.chunk_overlap = chunk_overlap  # Chunk overlap
        self.hf_embeddings = hf_embeddings  # HuggingFace embeddings for the document

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def process_pdf(self, file_path: str):
        """
        Load a single PDF file, split, chunk, and build a vector store using HF embeddings.
        Only rebuilds the vector store if the file has changed.
        """
        if not file_path.endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")

        # Generate a hash of the new uploaded file
        new_file_hash = get_file_hash(file_path)

        # Load previous file hash and compare to detect if rebuild is needed
        if os.path.exists(self.persist_dir) and os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                old_hash = f.read().strip()

            # If file hasn't changed, load existing vector store
            if old_hash == new_file_hash:
                logger.info("Same file detected. Loading existing vector store...")
                return Chroma(
                    persist_directory=self.persist_dir,
                    collection_name="pdf-collection",
                    embedding_function=self.hf_embeddings
                )
            else:
                logger.info("New file detected. Rebuilding vector store...")

        else:
            logger.info("No previous file found. Building vector store...")

        # Clear existing vector store and rebuild
        clear_vector_store(self.persist_dir)

        # Load and split PDF file
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        except Exception as e:
            logger.exception(f"Failed to load PDF {file_path}: {e}")
            raise ValueError("Could not process PDF. Please ensure it's a valid file.")

        # Add metadata to each chunk
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["embedding_model"] = "huggingface"

        # Split documents into chunks
        try:
            chunks = self.text_splitter.split_documents(docs)
        except Exception as e:
            logger.exception(f"Failed to split document into chunks: {e}")
            raise RuntimeError("An error occurred while splitting the document into chunks.")

        # Build new Chroma vector store
        try:
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.hf_embeddings,
                collection_name="pdf-collection",
                persist_directory=self.persist_dir
            )
        except Exception as e:
            logger.exception(f"Failed to create a vector store: {e}")
            raise RuntimeError("An error occurred while creating the vector store.")

        # Save new file hash to disk for future comparisons
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(METADATA_FILE, "w") as f:
            f.write(new_file_hash)

        logger.info("Vector store successfully created.")
        return vector_store
