from huggingface_hub import snapshot_download
import os

nsfw_model_name = "Falconsai/nsfw_image_detection"
nsfw_dir = os.getenv("MODEL_PATH_NSFW", "./model_nsfw")

if not os.path.exists(nsfw_dir):
    os.makedirs(nsfw_dir)
print(f"Downloading model {nsfw_model_name} to {nsfw_dir}...")
snapshot_download(repo_id=nsfw_model_name, local_dir=nsfw_dir)
print(f"Model files downloaded successfully: {nsfw_model_name}")
