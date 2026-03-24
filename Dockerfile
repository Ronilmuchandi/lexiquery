# =============================================================
# FILE: Dockerfile
# PURPOSE: Package LexiQuery into a container so it can run
#          anywhere — any computer, any server, any cloud
#
# SIMPLE ANALOGY:
# Think of Docker like a lunchbox. Everything your app needs
# (Python, packages, code) is packed inside. Anyone can open
# the lunchbox and the app just works — no "it works on my
# machine" problems.
#
# HOW TO BUILD:  docker build -t lexiquery .
# HOW TO RUN:    docker run -p 8501:8501 lexiquery
# =============================================================

# Start with Python 3.11 as our base
# "slim" means minimal size — no unnecessary extras
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (for Docker layer caching)
# This means if code changes but requirements don't,
# Docker won't reinstall packages every time
COPY requirements.txt .

# Install all Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set Python path so src module is found
ENV PYTHONPATH=/app

# Expose port 8501 (Streamlit's default port)
EXPOSE 8501

# Command to run when container starts
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]