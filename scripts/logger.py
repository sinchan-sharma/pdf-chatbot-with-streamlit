import logging
import os
import streamlit as st

LOG_DIR = "logs"
LOG_FILE_NAME = "pdf_rag.log"
LOGGER_NAME = "pdf_rag"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

def get_logger():
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    # Prevent log propagation to root logger (avoids double-logging)
    logger.propagate = False

    if logger.handlers:
        return logger

    # Logging formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Always add a console handler (for both local runs & inside Docker)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Log to file ONLY if the app is not being run inside Docker
    if os.getenv("RUNNING_IN_DOCKER") != "true":

        # Create a new folder if it doesn't already exist
        os.makedirs(LOG_DIR, exist_ok=True)

        # Clear file once per Streamlit session
        if "log_initialized" not in st.session_state:
            with open(LOG_FILE_PATH, "w"):
                pass
            st.session_state["log_initialized"] = True

        fh = logging.FileHandler(LOG_FILE_PATH, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
