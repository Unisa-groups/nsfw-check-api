# AGENTS.md

FastAPI service that classifies an uploaded image as NSFW or not, using the
`Falconsai/nsfw_image_detection` ViT model.

## Layout

- `src/nsfw_check_api/main.py` — the whole app (endpoints + inference)
- `tests/` — pytest suite
- `model/` — model weights, git-ignored, mounted as a volume in Docker
- `docs/consuming-this-api.md` — integration guide for projects that call this service

## Commands

Dependency manager is **PDM** (`pdm.lock` is committed).

```bash
pdm install                       # sync venv from the lock
pdm run ruff check .              # lint (CI gate)
pdm run pytest                    # full suite (needs model/ populated)
pdm run pytest -m "not needs_model"   # subset that needs no model
pdm run uvicorn nsfw_check_api.main:app --port 15000
```

After changing dependencies: `pdm add ... ` / edit `pyproject.toml`, then
`pdm lock` and commit `pdm.lock`.

## The model

- `_load_model` is `@lru_cache`d and warmed in the FastAPI `lifespan` on
  startup. Tests use a module-level `TestClient` (no `with`), so lifespan
  doesn't fire and importing the module stays model-free.
- Tests set `HF_HUB_OFFLINE=1` in `tests/conftest.py` — a test run must never
  download the model. Tests that actually need it are marked `needs_model` and
  expect the files already in `./model`; CI restores `model/` from an
  `actions/cache` and fetches it once on a miss (outside pytest, so offline
  mode doesn't apply).
- `ensure_models_exist()` pulls only `*.json` / `*.safetensors`
  (`_MODEL_FILE_PATTERNS`, ~330 MB) — the HF repo also ships `optimizer.pt`
  (655 MB), a yolo variant, and a redundant `pytorch_model.bin`. It's guarded
  by a `FileLock` so that with `WEB_CONCURRENCY>1` only one worker downloads
  and a partial download can't be mistaken for a complete one.
- `torch` is pinned to a CPU-only wheel (`[[tool.pdm.source]]`) to keep the
  image small. Don't switch to `AutoImageProcessor` — it pulls in torchvision;
  `ViTImageProcessorPil` is deliberate.

## Concurrency

- One `uvicorn` process has one GIL, so parallelism comes from **worker
  processes**: `WEB_CONCURRENCY` (Docker default `2`) sets the count; each
  worker loads its own ~350 MB model copy.
- `OMP_NUM_THREADS=1` (Docker) keeps each worker's torch to one thread so N
  workers don't oversubscribe the cores.
- `MAX_INFLIGHT` (default `2`) is a per-worker `asyncio.Semaphore` around
  inference. When it's full, `/nsfw_check` returns `503` + `Retry-After: 1`
  instead of queueing. System capacity ≈ `WEB_CONCURRENCY × MAX_INFLIGHT`.

## `/nsfw_check` response

```json
{"is_nsfw": true, "nsfw_probability": 0.9982,
 "meta": {"inference_ms": 41.2, "total_ms": 47.9, "threshold": 0.5,
          "model": "...", "image": {"width": ..., "height": ..., "format": ...,
          "mode": ..., "bytes": ...}, "worker_pid": 7}}
```
`meta` is diagnostic and cheap to compute; the top level is the answer.

Each worker logs `... ready: model=... pid=...` once on startup and
`nsfw_check: is_nsfw=... probability=...` per image, via the `uvicorn.error`
logger so the lines share uvicorn's output and formatting.

## Docker

`Dockerfile` builds on `python:3.12-slim` via `pdm install`. Both compose
files define a `healthcheck` hitting `/heartbeat` (podman-compose doesn't
inherit an image `HEALTHCHECK`). `CI` pushes to `ghcr.io/<repo>` on `main`.
The model is not baked in — mount it at `/usr/src/app/model`.

## Commits

Use [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

- **type**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
  `build`, `ci`, `chore`, `revert`
- Breaking change: `!` after type/scope (`feat!: ...`) or a `BREAKING CHANGE:`
  footer.
- Description: imperative mood, lower-case, no trailing period.

Examples: `fix: reject uploads larger than MAX_UPLOAD_BYTES`,
`ci: build the image on main`, `refactor(main): load the model lazily`.
