# Use official Python slim image to keep it light
FROM python:3.11-slim

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    RUNNING_IN_DOCKER=true

# Set working directory
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install torch CPU wheel first explicitly
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# Remove torch from requirements.txt to avoid reinstallation
RUN sed -i '/torch/d' requirements.txt

# Upgrade pip and install the rest of dependencies without torch
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy the rest of the app source code
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "scripts/app.py", "--server.port=8501", "--server.address=0.0.0.0"]


# Note: to build Docker image, run the following in the terminal:
# docker build -t streamlit-chatbot . 

# -t streamlit-chatbot names the image
# . tells Docker to use the current directory as context

# To run the app, run the following in the terminal:
# docker run --rm -p 8501:8501 --env-file .env streamlit-chatbot

# Or run docker run -p 8501:8501 -v $(pwd)/.env:/app/.env streamlit-chatbot
# Or run docker run -p 8501:8501 -e GOOGLE_API_KEY=xxx -e GROQ_API_KEY=yyy your-image

# This will do the following:
# Bind your host’s port 8501 to the container’s port 8501
# Inject your .env variables at runtime
# Automatically remove the container after you stop it#
# Then visit http://localhost:8501 in your browser.
# Or you can visit http://127.0.0.1:8501 instead (effectively the same thing)
