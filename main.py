from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image, UnidentifiedImageError
import io

# Load model and image processor from local directory
model_name = "Falconsai/nsfw_image_detection"
model_path = "./model"
image_processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)

app = FastAPI()

def is_nsfw(image):
    # Preprocess the image
    inputs = image_processor(images=image, return_tensors="pt")
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    nsfw_prob = probabilities[0][1].item()  # Index 1 is NSFW
    return nsfw_prob > 0.5, nsfw_prob

@app.post("/check_nsfw")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8123)
