import io
from PIL import Image
from fastapi.testclient import TestClient
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

def test_check_nsfw_endpoint():
    # Create a small blank image in memory
    image = Image.new('RGB', (100, 100), color='blue')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Use TestClient to send a POST request with the file
    response = client.post(
        "/check_nsfw",
        files={"file": ("test.png", img_byte_arr, "image/png")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert data["filename"] == "test.png"
    assert "is_nsfw" in data
    assert not data["is_nsfw"]
    assert "nsfw_probability" in data
    assert data["nsfw_probability"] < 0.5

def test_check_nsfw_invalid_file():
    # Test with a non-image file
    response = client.post(
        "/check_nsfw",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    )
    
    # The API should handle this and return 400
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file"

def test_test_endpoint():
    response = client.get("/nsfw_test")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "NSFW Check Test" in response.text
    assert '<form id="upload-form">' in response.text

def test_ocr_check_endpoint():
    # Create a small blank image with some "text" (though we are just mocking the model response if needed, 
    # but here it's an end-to-end test so we expect whatever the model returns for a blank image)
    image = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Use TestClient to send a POST request with the file
    response = client.post(
        "/ocr_check",
        files={"file": ("test_ocr.png", img_byte_arr, "image/png")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert data["filename"] == "test_ocr.png"
    assert "ocr_text" in data
    assert isinstance(data["ocr_text"], str)

def test_ocr_test_endpoint():
    response = client.get("/ocr_test")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "OCR Check Test" in response.text
    assert '<form id="upload-form">' in response.text
