import asyncio
import io
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForImageClassification, ViTImageProcessorPil

model_path = os.getenv("MODEL_PATH", "./model")
model_name = os.getenv("MODEL_NAME", "Falconsai/nsfw_image_detection")
max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
nsfw_threshold = float(os.getenv("NSFW_THRESHOLD", "0.5"))
# Max inferences running at once in this worker; further requests get 503 rather than piling up
max_inflight = int(os.getenv("MAX_INFLIGHT", "2"))

# config + processor + safetensors weights only: skips optimizer.pt (655 MB),
# the bundled yolo .pt, and the redundant pickle pytorch_model.bin.
# Add "*.bin" here if you point MODEL_NAME at a model with no safetensors.
_MODEL_FILE_PATTERNS = ["*.json", "*.safetensors"]

def _model_present():
    # both a config and a weights file - config.json alone means a download is
    # still in flight (it's tiny and lands first)
    have = set(os.listdir(model_path)) if os.path.isdir(model_path) else set()
    return "config.json" in have and bool(have & {"model.safetensors", "pytorch_model.bin"})

def ensure_models_exist():
    if _model_present():
        return
    from filelock import FileLock
    os.makedirs(model_path, exist_ok=True)
    # serialize across uvicorn workers: only one downloads, the rest wait then find it
    with FileLock(os.path.join(model_path, ".download.lock")):
        if _model_present():
            return
        from huggingface_hub import snapshot_download
        print(f"Downloading model {model_name} to {model_path}...")
        snapshot_download(
            repo_id=model_name,
            local_dir=model_path,
            allow_patterns=_MODEL_FILE_PATTERNS,
        )
        print(f"Model files downloaded successfully: {model_name}")

@lru_cache(maxsize=1)
def _load_model():
    # Lazy so importing this module (e.g. in tests) doesn't pull in the model.
    # ViTImageProcessorPil is the PIL-only processor, so torchvision isn't required.
    ensure_models_exist()
    processor = ViTImageProcessorPil.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    # Resolve the NSFW class index from the config instead of assuming it is 1
    # (this model stores label2id values as strings, hence int())
    label2id = {label.lower(): int(idx) for label, idx in model.config.label2id.items()}
    return processor, model, label2id["nsfw"]

@asynccontextmanager
async def lifespan(_app):
    # Warm the model before serving so the first request isn't slow and a burst
    # of concurrent first requests doesn't each trigger a load.
    _load_model()
    yield

app = FastAPI(lifespan=lifespan)

inference_slots = asyncio.Semaphore(max_inflight)

def is_nsfw(image):
    processor, model, nsfw_index = _load_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    nsfw_prob = probabilities[0][nsfw_index].item()
    return nsfw_prob > nsfw_threshold, nsfw_prob

@app.get("/heartbeat")
async def heartbeat():
    return {"status": "alive"}


@app.post("/nsfw_check")
async def check_nsfw(file: UploadFile = File(...)):
    started = time.perf_counter()
    if file.size is not None and file.size > max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image too large")
    contents = await file.read()
    if len(contents) > max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image too large")
    try:
        # convert() forces a full decode, so truncated/corrupt/bomb images fail here, not mid-inference
        with Image.open(io.BytesIO(contents)) as img:
            image_meta = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "bytes": len(contents),
            }
            image = img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image file") from None

    if inference_slots.locked():
        raise HTTPException(
            status_code=503,
            detail="Server busy, retry shortly",
            headers={"Retry-After": "1"},
        )
    async with inference_slots:
        inference_started = time.perf_counter()
        is_nsfw_bool, prob = await run_in_threadpool(is_nsfw, image)
        inference_ms = (time.perf_counter() - inference_started) * 1000

    return {
        "is_nsfw": is_nsfw_bool,
        "nsfw_probability": round(prob, 4),
        "meta": {
            "inference_ms": round(inference_ms, 1),
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
            "threshold": nsfw_threshold,
            "model": model_name,
            "image": image_meta,
            "worker_pid": os.getpid(),
        },
    }


@app.get("/nsfw_test")
async def test_form():
    content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NSFW Check Test</title>
        <style>
            body { font-family: sans-serif; margin: 2rem; }
            .result { margin-top: 1rem; padding: 1rem; border: 1px solid #ccc; border-radius: 4px; display: none; }
        </style>
    </head>
    <body>
        <h1>NSFW Check Test</h1>
        <form id="upload-form">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Upload and Check</button>
        </form>
        <div id="result" class="result"></div>

        <script>
            const form = document.getElementById('upload-form');
            const resultDiv = document.getElementById('result');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(form);
                resultDiv.style.display = 'block';
                resultDiv.innerText = 'Analyzing...';

                try {
                    const response = await fetch('/nsfw_check', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    resultDiv.innerText = JSON.stringify(result, null, 2);
                    resultDiv.style.whiteSpace = 'pre-wrap';
                } catch (error) {
                    resultDiv.innerText = 'Error: ' + error.message;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=15000)
