# AGENTS.md

FastAPI service that classifies an uploaded image as NSFW or not, using the
`Falconsai/nsfw_image_detection` ViT model.

## Layout

- `src/nsfw_check_api/main.py` — the whole app (endpoints + inference)
- `tests/` — pytest suite
- `model/` — model weights, git-ignored, mounted as a volume in Docker

## Commands

Dependency manager is **PDM** (`pdm.lock` is committed).

```bash
pdm install                       # sync venv from the lock
pdm run pytest                    # full suite (needs model/ populated)
pdm run pytest -m "not needs_model"   # what CI runs; no model required
pdm run uvicorn nsfw_check_api.main:app --port 30000
```

After changing dependencies: `pdm add ... ` / edit `pyproject.toml`, then
`pdm lock` and commit `pdm.lock`.

## The model

- Loaded lazily on the first `/nsfw_check` request (`_load_model`), not at import.
- Tests set `HF_HUB_OFFLINE=1` in `tests/conftest.py` — a test run must never
  download the model. Tests that actually need it are marked `needs_model` and
  expect the files already in `./model`.
- `torch` is pinned to a CPU-only wheel (`[[tool.pdm.source]]`) to keep the
  image small. Don't switch to `AutoImageProcessor` — it pulls in torchvision;
  `ViTImageProcessorPil` is deliberate.

## Docker

`Dockerfile` builds on `python:3.12-slim` via `pdm install`. `CI` pushes to
`ghcr.io/<repo>` on `main`. The model is not baked in — mount it at
`/usr/src/app/model`.

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
