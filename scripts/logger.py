import logging
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

name = "pdf_rag"
logger = logging.getLogger(name)
logger.setLevel(logging.DEBUG)  # or INFO in production

# Prevent adding multiple handlers during Streamlit reruns
# This prevents the same outputs from showing up repeatedly in the terminal
if not logger.hasHandlers():
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(f"{log_dir}/{name}.log", mode="w")
    fh.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

def get_logger():
    return logger
