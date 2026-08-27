import os

# Never reach out to the Hugging Face Hub during tests - the model must already
# be present in ./model (tests that need it are marked `needs_model`).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
