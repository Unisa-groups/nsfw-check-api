# Consuming the NSFW check API

Drop this into the consuming project (e.g. its `AGENTS.md`) so whoever works
there knows how to call the service. Adjust the base URL if it isn't running
on the same host.

---

## nsfw-check-api

A small HTTP service that classifies an image as NSFW or not. Screen
user-supplied images through it before storing or serving them.

**Base URL:** `http://localhost:30000` (the container publishes port 30000).
No authentication.

### `POST /nsfw_check`

`multipart/form-data`, one field:

| field | value |
|-------|-------|
| `file` | the image bytes (JPEG/PNG/WebP/… anything Pillow decodes). Max 10 MB. |

**200** — JSON:

```json
{
  "is_nsfw": true,
  "nsfw_probability": 0.9982,
  "meta": {
    "inference_ms": 41.2,
    "total_ms": 47.9,
    "threshold": 0.5,
    "model": "Falconsai/nsfw_image_detection",
    "image": {"width": 1024, "height": 768, "format": "JPEG", "mode": "RGB", "bytes": 84213},
    "worker_pid": 7
  }
}
```

Use `is_nsfw` for the yes/no decision. `nsfw_probability` (0–1) is the raw
score if you want your own cutoff; `meta.threshold` is the server-side one.
`meta` is diagnostic — safe to log, safe to ignore.

### Errors

| status | meaning | what the consumer should do |
|--------|---------|-----------------------------|
| `400` | `{"detail": "Invalid image file"}` — not a decodable image | reject the upload, tell the user |
| `413` | `{"detail": "Image too large"}` — over 10 MB | reject the upload, tell the user |
| `503` | `{"detail": "Server busy, retry shortly"}` + `Retry-After` header | retry with backoff — **do not** skip the check |

### `GET /heartbeat`

`{"status": "alive"}` — use for readiness/liveness checks.

---

## Python client

```python
import time
import httpx

NSFW_API = "http://localhost:30000"


def check_nsfw(image_bytes: bytes, filename: str = "image", *, retries: int = 3) -> dict:
    """Screen an image. Returns the full response dict (is_nsfw, nsfw_probability, meta).

    Raises httpx.HTTPStatusError on 400 (undecodable) / 413 (too large),
    and RuntimeError if the service stays saturated after `retries`.
    """
    for attempt in range(retries + 1):
        resp = httpx.post(
            f"{NSFW_API}/nsfw_check",
            files={"file": (filename, image_bytes)},
            timeout=30,
        )
        if resp.status_code == 503:
            if attempt == retries:
                raise RuntimeError("nsfw-check-api is saturated")
            time.sleep(float(resp.headers.get("Retry-After", "1")) * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()


# usage
result = check_nsfw(open("photo.jpg", "rb").read(), "photo.jpg")
if result["is_nsfw"]:
    reject(...)
```

Needs `httpx` (or swap in `requests` — same `files=` call). Don't hand-roll
the request elsewhere; go through this one function so the 503 retry stays in
one place.

## curl

```bash
curl -sS -F "file=@photo.jpg" http://localhost:30000/nsfw_check
```
