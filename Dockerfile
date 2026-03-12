FROM astral/uv:python3.12-bookworm-slim

WORKDIR /usr/src/app
ENV UV_COMPILE_BYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/usr/src/app/.venv
ENV PATH="/usr/src/app/.venv/bin:$PATH"
ENV MODEL_PATH=/usr/src/app/model

COPY . ./
RUN uv sync --no-cache --no-install-project --no-dev
RUN uv run download_model.py

EXPOSE 8123
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8123"]
