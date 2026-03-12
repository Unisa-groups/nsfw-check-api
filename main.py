from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image, UnidentifiedImageError
import os
import io

# Load model and image processor from local directory
nsfw_model_path = os.getenv("MODEL_PATH_NSFW", "./model_nsfw")
nsfw_model_name = "Falconsai/nsfw_image_detection"
nsfw_image_processor = AutoImageProcessor.from_pretrained(nsfw_model_path, local_files_only=True)
nsfw_model = AutoModelForImageClassification.from_pretrained(nsfw_model_path, local_files_only=True)

ocr_model_name = "zai-org/GLM-OCR"
ocr_model_path = os.getenv("MODEL_PATH_OCR", "./model_ocr")
ocr_tokenizer = AutoTokenizer.from_pretrained(ocr_model_path, local_files_only=True)
ocr_model = AutoModelForImageTextToText.from_pretrained(ocr_model_path, local_files_only=True)
app = FastAPI()

def is_nsfw(image):
    # Preprocess the image
    inputs = nsfw_image_processor(images=image, return_tensors="pt")
    # Predict
    with torch.no_grad():
        outputs = nsfw_model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    nsfw_prob = probabilities[0][1].item()  # Index 1 is NSFW
    return nsfw_prob > 0.5, nsfw_prob

def perform_ocr(image):
    # Preprocess the image
    inputs = ocr_tokenizer(image, return_tensors="pt")
    # Predict
    with torch.no_grad():
        outputs = ocr_model.generate(**inputs, max_new_tokens=512)
    # Decode result
    text = ocr_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return text

@app.post("/nsfw_check")
async def check_nsfw(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    is_nsfw_bool, prob = is_nsfw(image)
    return JSONResponse(
        content={
            "filename": file.filename,
            "is_nsfw": is_nsfw_bool,
            "nsfw_probability": prob
        }
    )

@app.post("/ocr_check")
async def check_ocr(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    text = perform_ocr(image)
    return JSONResponse(
        content={
            "filename": file.filename,
            "ocr_text": text
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
                    const response = await fetch('/check_nsfw', {
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

@app.get("/ocr_test")
async def ocr_test_form():
    content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OCR Check Test</title>
        <style>
            body { font-family: sans-serif; margin: 2rem; }
            .result { margin-top: 1rem; padding: 1rem; border: 1px solid #ccc; border-radius: 4px; display: none; }
        </style>
    </head>
    <body>
        <h1>OCR Check Test</h1>
        <form id="upload-form">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Upload and Perform OCR</button>
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
                    const response = await fetch('/ocr_check', {
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
