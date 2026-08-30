FROM python:3.12-slim

# Set up non-root user for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency manifests first for Docker layer caching
COPY --chown=user:user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy remaining project files
COPY --chown=user:user . .

# Expose Hugging Face default port
ENV PORT=7860
EXPOSE 7860

# Run FastAPI server
CMD ["uv", "run", "uvicorn", "src.gui.server:app", "--host", "0.0.0.0", "--port", "7860"]
