from huggingface_hub import snapshot_download
import os

nsfw_model_name = "Falconsai/nsfw_image_detection"
nsfw_dir = os.getenv("MODEL_PATH_NSFW", "./model_nsfw")

if not os.path.exists(nsfw_dir):
    os.makedirs(nsfw_dir)
print(f"Downloading model {nsfw_model_name} to {nsfw_dir}...")
snapshot_download(repo_id=nsfw_model_name, local_dir=nsfw_dir)
print(f"Model files downloaded successfully: {nsfw_model_name}")

ocr_model_name = "zai-org/GLM-OCR"
ocr_dir = os.getenv("MODEL_PATH_OCR", "./model_ocr")
if not os.path.exists(ocr_dir):
    os.makedirs(ocr_dir)
print(f"Downloading model {ocr_model_name} to {ocr_dir}...")
snapshot_download(repo_id=ocr_model_name, local_dir=ocr_dir)
print(f"Model files downloaded successfully: {ocr_model_name}")