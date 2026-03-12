# Stage 1: Build stage
FROM astral/uv:python3.12-bookworm-slim as builder

WORKDIR /usr/src/app
ENV UV_COMPILE_BYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/usr/src/app/.venv
ENV PATH="/usr/src/app/.venv/bin:$PATH"

# Copy only requirements first to leverage Docker cache
COPY pyproject.toml ./
RUN uv sync --no-cache --no-install-project --no-dev

# Copy the rest of the application
COPY . ./

# Download models (or build artifacts)
RUN uv run /usr/src/app/download_model.py

# ---

# Stage 2: Runtime stage
FROM astral/uv:python3.12-bookworm-slim

WORKDIR /usr/src/app
ENV UV_COMPILE_BYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/usr/src/app/.venv
ENV PATH="/usr/src/app/.venv/bin:$PATH"
ENV MODEL_PATH_NSFW=/usr/src/app/model_nsfw
ENV MODEL_PATH_OCR=/usr/src/app/model_ocr

# Copy only the virtual environment and necessary files from the builder
COPY --from=builder /usr/src/app/.venv /usr/src/app/.venv
COPY --from=builder /usr/src/app/model_nsfw /usr/src/app/model_nsfw
COPY --from=builder /usr/src/app/model_ocr /usr/src/app/model_ocr
COPY --from=builder /usr/src/app/main.py /usr/src/app/main.py

EXPOSE 8123
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8123"]
