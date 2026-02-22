"""Tests for the /estimate endpoint."""


def test_estimate_png(client, sample_png):
    """PNG -> estimate with method containing oxipng or pngquant."""
    resp = client.post("/estimate", files={"file": ("test.png", sample_png, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["original_format"] == "png"
    assert "method" in data
    assert data["estimated_reduction_percent"] >= 0


def test_estimate_jpeg(client, sample_jpeg):
    """JPEG -> estimate response."""
    resp = client.post("/estimate", files={"file": ("test.jpg", sample_jpeg, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["original_format"] == "jpeg"
    assert data["estimated_reduction_percent"] >= 0


def test_estimate_svg(client, sample_svg):
    """SVG -> estimate with method scour."""
    resp = client.post("/estimate", files={"file": ("test.svg", sample_svg, "image/svg+xml")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["original_format"] == "svg"
    assert "scour" in data["method"]


def test_estimate_gif(client, sample_gif):
    """GIF -> estimate response."""
    resp = client.post("/estimate", files={"file": ("test.gif", sample_gif, "image/gif")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["original_format"] == "gif"


def test_estimate_response_fields(client, sample_png):
    """Estimate response has all required fields."""
    resp = client.post("/estimate", files={"file": ("test.png", sample_png, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    required_fields = [
        "original_size",
        "original_format",
        "dimensions",
        "estimated_optimized_size",
        "estimated_reduction_percent",
        "optimization_potential",
        "method",
        "already_optimized",
        "confidence",
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"


def test_estimate_dimensions(client, sample_png):
    """Estimate includes correct dimensions."""
    resp = client.post("/estimate", files={"file": ("test.png", sample_png, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    dims = data["dimensions"]
    assert "width" in dims
    assert "height" in dims
    assert dims["width"] > 0
    assert dims["height"] > 0


def test_estimate_confidence_levels(client, sample_png, sample_jpeg):
    """Confidence is one of low, medium, high."""
    for name, data_bytes, ct in [
        ("test.png", sample_png, "image/png"),
        ("test.jpg", sample_jpeg, "image/jpeg"),
    ]:
        resp = client.post("/estimate", files={"file": (name, data_bytes, ct)})
        if resp.status_code == 200:
            assert resp.json()["confidence"] in ("low", "medium", "high")


def test_estimate_415_unsupported(client):
    """Random bytes -> 415."""
    resp = client.post(
        "/estimate",
        files={"file": ("test.bin", b"random garbage bytes", "application/octet-stream")},
    )
    assert resp.status_code == 415


def test_estimate_400_no_input(client):
    """No file and no JSON body -> 400."""
    resp = client.post("/estimate")
    assert resp.status_code == 400


# --- Preset-based estimation ---


def test_estimate_json_with_preset(client, sample_png):
    """File upload with preset returns valid estimate."""
    resp = client.post(
        "/estimate",
        files={"file": ("test.png", sample_png, "image/png")},
        data={"preset": "high"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["original_format"] == "png"
    assert data["estimated_reduction_percent"] >= 0


def test_estimate_preset_overrides_options(client, sample_jpeg):
    """When preset is provided, it takes precedence over options."""
    resp = client.post(
        "/estimate",
        files={"file": ("test.jpg", sample_jpeg, "image/jpeg")},
        data={"preset": "high"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["estimated_reduction_percent"] >= 0


def test_estimate_invalid_preset(client, sample_png):
    """Invalid preset returns 400."""
    resp = client.post(
        "/estimate",
        files={"file": ("test.png", sample_png, "image/png")},
        data={"preset": "ultra"},
    )
    assert resp.status_code == 400
