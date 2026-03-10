from huggingface_hub import snapshot_download
import os

model_name = "Falconsai/nsfw_image_detection"
local_dir = "./model"

if not os.path.exists(local_dir):
    os.makedirs(local_dir)

print(f"Downloading model {model_name} to {local_dir}...")
snapshot_download(repo_id=model_name, local_dir=local_dir)
print("Model files downloaded successfully.")
