FROM python:3.12-slim-bookworm

WORKDIR /usr/src/app
ENV PATH="/usr/src/app/.venv/bin:$PATH"
ENV MODEL_PATH="/usr/src/app/model"
ENV MODEL_NAME="Falconsai/nsfw_image_detection"
ENV PDM_CHECK_UPDATE=false

RUN pip install --no-cache-dir pdm

# Deps first so this layer is cached until the lockfile changes
COPY pyproject.toml pdm.lock ./
RUN pdm install --prod --no-self --frozen-lockfile && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

COPY . ./
RUN pdm install --prod --frozen-lockfile

EXPOSE 8123
CMD ["uvicorn", "nsfw_check_api.main:app", "--host", "0.0.0.0", "--port", "8123"]
