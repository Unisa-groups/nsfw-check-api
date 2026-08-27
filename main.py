from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.concurrency import run_in_threadpool
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image, UnidentifiedImageError
import os
import io

# Load model and image processor from local directory
model_path = os.getenv("MODEL_PATH", "./model")
model_name = os.getenv("MODEL_NAME", "Falconsai/nsfw_image_detection")
max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

def ensure_models_exist():
    # Check for config.json as an indicator that the model is present
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        from huggingface_hub import snapshot_download
        print(f"Downloading model {model_name} to {model_path}...")
        os.makedirs(model_path, exist_ok=True)
        snapshot_download(repo_id=model_name, local_dir=model_path)
        print(f"Model files downloaded successfully: {model_name}")

ensure_models_exist()

image_processor = AutoImageProcessor.from_pretrained(model_path)
nsfw_model = AutoModelForImageClassification.from_pretrained(model_path)

app = FastAPI()

def is_nsfw(image):
    # Preprocess the image
    inputs = image_processor(images=image, return_tensors="pt")
    # Predict
    with torch.no_grad():
        outputs = nsfw_model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    nsfw_prob = probabilities[0][1].item()  # Index 1 is NSFW
    return nsfw_prob > 0.5, nsfw_prob

@app.get("/heartbeat")
async def heartbeat():
    return {"status": "alive"}


@app.post("/nsfw_check")
async def check_nsfw(file: UploadFile = File(...)):
    if file.size is not None and file.size > max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image too large")
    contents = await file.read()
    if len(contents) > max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image too large")
    try:
        image = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    is_nsfw_bool, prob = await run_in_threadpool(is_nsfw, image)
    prob = round(prob, 4)
    return JSONResponse(
        content={
            "is_nsfw": is_nsfw_bool,
            "nsfw_probability": prob
        }
    )


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
    uvicorn.run(app, host="0.0.0.0", port=8123)
