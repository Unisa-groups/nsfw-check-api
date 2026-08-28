import asyncio
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

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

def test_logs_request_line_and_rejection(caplog):
    with caplog.at_level("INFO", logger="uvicorn.error"):
        client.post(
            "/nsfw_check",
            files={"file": ("bad.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
    msgs = [r.message for r in caplog.records]
    assert any("nsfw_check: request file=" in m for m in msgs)
    assert any("400" in m and "undecodable" in m for m in msgs)


def test_logs_busy_rejection(caplog, monkeypatch):
    monkeypatch.setattr(main, "inference_slots", asyncio.Semaphore(0))
    image = Image.new("RGB", (50, 50), color="red")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    with caplog.at_level("INFO", logger="uvicorn.error"):
        client.post("/nsfw_check", files={"file": ("busy.png", buf, "image/png")})
    assert any("503" in r.message and "busy" in r.message for r in caplog.records)


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

@pytest.mark.needs_model
def test_logs_on_startup(caplog):
    with caplog.at_level("INFO", logger="uvicorn.error"), TestClient(app):
        pass
    assert any("ready" in r.message for r in caplog.records)

@pytest.mark.needs_model
def test_logs_result_per_image(caplog):
    image = Image.new('RGB', (30, 30), color='red')
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    with caplog.at_level("INFO", logger="uvicorn.error"):
        client.post("/nsfw_check", files={"file": ("x.png", buf, "image/png")})
    assert any("is_nsfw=" in r.message for r in caplog.records)

def test_ensure_models_exist_only_fetches_inference_files(monkeypatch, tmp_path):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"x")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(main, "model_path", str(tmp_path))

    main.ensure_models_exist()

    patterns = captured["allow_patterns"]
    assert any(p.endswith(".safetensors") for p in patterns)
    # pickle weights (.bin), the 655 MB optimizer state and yolo variant (.pt) must be excluded
    assert not any(p.endswith((".bin", ".pt")) for p in patterns)

def test_ensure_models_exist_refetches_when_only_config_present(monkeypatch, tmp_path):
    # config.json alone is not "present" - a second worker seeing it mid-download
    # would otherwise skip and then fail to load missing weights
    (tmp_path / "config.json").write_text("{}")
    called = []

    def fake_snapshot_download(**kwargs):
        called.append(kwargs)
        (tmp_path / "model.safetensors").write_bytes(b"x")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(main, "model_path", str(tmp_path))

    main.ensure_models_exist()
    assert called

def test_ensure_models_exist_skips_when_weights_present(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x")

    def fail(**kwargs):
        raise AssertionError("snapshot_download should not run when the model is complete")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail)
    monkeypatch.setattr(main, "model_path", str(tmp_path))

    main.ensure_models_exist()

def test_test_endpoint():
    response = client.get("/nsfw_test")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "NSFW Check Test" in response.text
    assert '<form id="upload-form">' in response.text
    # Check if the fetch call points to the correct endpoint
    assert "fetch('/nsfw_check'" in response.text
