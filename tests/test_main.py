import asyncio
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from nsfw_check_api import main
from nsfw_check_api.main import app, is_nsfw

client = TestClient(app)

@pytest.mark.needs_model
def test_is_nsfw_function():
    # Create a small blank image for testing
    image = Image.new('RGB', (100, 100), color='red')
    is_nsfw_bool, prob = is_nsfw(image)
    
    # Red image should not be NSFW
    assert isinstance(is_nsfw_bool, bool)
    assert isinstance(prob, float)
    assert not is_nsfw_bool
    assert prob < 0.5

def test_heartbeat_endpoint():
    response = client.get("/heartbeat")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}

@pytest.mark.needs_model
def test_check_nsfw_endpoint():
    # Create a small blank image in memory
    image = Image.new('RGB', (100, 100), color='blue')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Use TestClient to send a POST request with the file
    response = client.post(
        "/nsfw_check",
        files={"file": ("test.png", img_byte_arr, "image/png")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "is_nsfw" in data
    assert not data["is_nsfw"]
    assert "nsfw_probability" in data
    assert data["nsfw_probability"] < 0.5

def test_check_nsfw_invalid_file():
    # Test with a non-image file
    response = client.post(
        "/nsfw_check",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    )

    # The API should handle this and return 400
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file"

def test_check_nsfw_truncated_image():
    # A file with a valid PNG header but a cut-off data stream: Image.open() succeeds,
    # the decode inside the endpoint does not.
    image = Image.new('RGB', (100, 100), color='green')
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    truncated = buf.getvalue()[:60]

    response = client.post(
        "/nsfw_check",
        files={"file": ("truncated.png", io.BytesIO(truncated), "image/png")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file"

@pytest.mark.needs_model
def test_check_nsfw_rgba_image():
    # Non-RGB images must be converted, not rejected or misclassified.
    image = Image.new('RGBA', (100, 100), color=(0, 0, 255, 128))
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)

    response = client.post(
        "/nsfw_check",
        files={"file": ("rgba.png", buf, "image/png")}
    )
    assert response.status_code == 200
    assert not response.json()["is_nsfw"]

def test_check_nsfw_too_large(monkeypatch):
    monkeypatch.setattr(main, "max_upload_bytes", 10)
    response = client.post(
        "/nsfw_check",
        files={"file": ("big.png", io.BytesIO(b"x" * 100), "image/png")}
    )
    assert response.status_code == 413

@pytest.mark.needs_model
def test_check_nsfw_meta():
    image = Image.new('RGBA', (120, 90), color=(0, 0, 255, 255))
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    png_bytes = buf.getvalue()

    data = client.post(
        "/nsfw_check",
        files={"file": ("m.png", io.BytesIO(png_bytes), "image/png")},
    ).json()

    meta = data["meta"]
    assert meta["inference_ms"] > 0
    assert meta["total_ms"] >= meta["inference_ms"]
    assert meta["threshold"] == 0.5
    assert meta["model"] == main.model_name
    assert meta["image"]["width"] == 120
    assert meta["image"]["height"] == 90
    assert meta["image"]["format"] == "PNG"
    assert meta["image"]["mode"] == "RGBA"
    assert meta["image"]["bytes"] == len(png_bytes)
    assert isinstance(meta["worker_pid"], int)

def test_check_nsfw_busy(monkeypatch):
    # No inference slots free -> reject immediately, don't queue
    monkeypatch.setattr(main, "inference_slots", asyncio.Semaphore(0))
    image = Image.new('RGB', (50, 50), color='red')
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)

    response = client.post(
        "/nsfw_check",
        files={"file": ("busy.png", buf, "image/png")},
    )
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"

@pytest.mark.needs_model
def test_lifespan_warms_model():
    main._load_model.cache_clear()
    assert main._load_model.cache_info().currsize == 0
    with TestClient(app):
        assert main._load_model.cache_info().currsize == 1

def test_test_endpoint():
    response = client.get("/nsfw_test")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "NSFW Check Test" in response.text
    assert '<form id="upload-form">' in response.text
    # Check if the fetch call points to the correct endpoint
    assert "fetch('/nsfw_check'" in response.text
