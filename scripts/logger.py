import logging
import os
import streamlit as st

LOG_DIR = "logs"
LOG_FILE_NAME = "pdf_rag.log"
LOGGER_NAME = "pdf_rag"

os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

def get_logger():
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    # Prevent log propagation to root logger (avoids double-logging)
    logger.propagate = False

    # Only configure handlers once (per Python process)
    if not logger.handlers:

        # Clear log file ONCE per Streamlit session
        if "log_initialized" not in st.session_state:
            with open(LOG_FILE_PATH, "w"):
                pass  # Just clear contents
            st.session_state["log_initialized"] = True

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(LOG_FILE_PATH, mode="a")
        fh.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Add handlers
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

