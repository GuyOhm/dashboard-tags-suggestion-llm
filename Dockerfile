# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for better caching)
COPY pyproject.toml uv.lock ./

# Create a non-root user and give ownership of the app directory
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -s /bin/bash -c "App User" appuser \
    && mkdir -p /home/appuser \
    && chown -R appuser:appuser /home/appuser \
    && chown -R appuser:appuser /app

# Install uv for the app user
USER appuser
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && /home/appuser/.local/bin/uv sync --frozen

# Add uv to PATH for subsequent commands
ENV PATH="/home/appuser/.local/bin:$PATH"

# Switch back to root temporarily to copy files and set permissions
USER root

# Copy application files and set ownership
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser .env.example ./

# Copy precomputed analysis JSON and samples df (if any)
COPY --chown=appuser:appuser analysis/ ./analysis/

# Switch back to non-root user
USER appuser

# Create .env from template if it doesn't exist
RUN if [ ! -f .env ]; then cp .env.example .env; fi

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the Streamlit application as non-root user
CMD ["/home/appuser/.local/bin/uv", "run", "streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
