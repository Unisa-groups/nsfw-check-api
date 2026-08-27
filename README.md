# nsfw-check-api

A small FastAPI service that classifies an uploaded image as NSFW or not,
using the [`Falconsai/nsfw_image_detection`](https://huggingface.co/Falconsai/nsfw_image_detection)
ViT model on CPU.

## Run

```bash
docker compose up --build
```

The model is not baked into the image — it downloads to `./model/` on first
start (mounted as a volume) and is reused after that. The API listens on
**http://localhost:15000**.

Prebuilt images: `ghcr.io/unisa-groups/nsfw-check-api` (pushed on every merge
to `main`).

## Use

```bash
curl -sS -F "file=@photo.jpg" http://localhost:15000/nsfw_check
```
```json
{"is_nsfw": false, "nsfw_probability": 0.0001, "meta": { ... }}
```

- `POST /nsfw_check` — multipart `file`, max 10 MB. `400` bad image, `413`
  too large, `503` + `Retry-After` when saturated.
- `GET /heartbeat` — `{"status": "alive"}`.

Full contract and a Python client: [`docs/consuming-this-api.md`](docs/consuming-this-api.md).

## Configuration

| env | default | |
|-----|---------|-|
| `NSFW_THRESHOLD` | `0.5` | probability above which `is_nsfw` is true |
| `MAX_UPLOAD_BYTES` | `10485760` | reject larger uploads with `413` |
| `WEB_CONCURRENCY` | `2` | uvicorn worker processes (each ~350 MB for its model copy) |
| `MAX_INFLIGHT` | `2` | concurrent inferences per worker before `503` |
| `MODEL_NAME` / `MODEL_PATH` | Falconsai / `./model` | which model, and where it lives |

## Development

See [`AGENTS.md`](AGENTS.md).
