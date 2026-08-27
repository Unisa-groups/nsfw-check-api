FROM python:3.12-slim-bookworm

WORKDIR /usr/src/app
ENV MODEL_PATH=/usr/src/app/model \
    MODEL_NAME=Falconsai/nsfw_image_detection \
    PDM_CHECK_UPDATE=false \
    PATH=/usr/src/app/.venv/bin:$PATH

RUN pip install --no-cache-dir pdm

# Locked prod deps first so this layer is cached until pyproject/pdm.lock change.
# pdm replays the lock (incl. the CPU-only torch source) into an in-project .venv.
COPY pyproject.toml pdm.lock ./
RUN pdm install --prod --no-self --frozen-lockfile && rm -rf /root/.cache

COPY src ./src
RUN pdm install --prod --frozen-lockfile && rm -rf /root/.cache

# Runtime knobs (see AGENTS.md > Concurrency); safe to override per host
ENV WEB_CONCURRENCY=2 \
    OMP_NUM_THREADS=1

EXPOSE 15000
CMD ["uvicorn", "nsfw_check_api.main:app", "--host", "0.0.0.0", "--port", "15000"]
