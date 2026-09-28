# Use a minimal base image with Python 3.13 installed
FROM python:3.13.5-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies first
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Create a non-root user for security (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy all files from the current directory on the host to the container's app directory
COPY --chown=user . $HOME/app

# Define the command to run the Streamlit app on port 7860 and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
