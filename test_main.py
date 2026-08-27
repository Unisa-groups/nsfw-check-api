import io
from PIL import Image
from fastapi.testclient import TestClient
import main
from main import app, is_nsfw

client = TestClient(app)

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

def test_test_endpoint():
    response = client.get("/nsfw_test")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "NSFW Check Test" in response.text
    assert '<form id="upload-form">' in response.text
    # Check if the fetch call points to the correct endpoint
    assert "fetch('/nsfw_check'" in response.text
