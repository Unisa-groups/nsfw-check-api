FROM astral/uv:python3.12-bookworm-slim

WORKDIR /usr/src/app
ENV UV_COMPILE_BYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/usr/src/app/.venv
ENV PATH="/usr/src/app/.venv/bin:$PATH"
ENV MODEL_PATH="/usr/src/app/model"
ENV MODEL_NAME="Falconsai/nsfw_image_detection"

COPY pyproject.toml ./
RUN uv sync --no-cache --no-install-project --no-dev && \
    rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/*

COPY . ./

EXPOSE 8123
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8123"]
