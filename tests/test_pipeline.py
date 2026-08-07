import os
import sys
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_captions, split_train_val  # noqa: E402

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
DUMMY_CAPTIONS_PATH = os.path.join(FIXTURES_DIR, "dummy_captions.txt")


# ---------- data_loader.py tests (using small dummy data, no real dataset needed) ----------


def test_load_captions_returns_dataframe():
    df = load_captions(DUMMY_CAPTIONS_PATH)
    assert df is not None
    assert len(df) > 0
    assert "image_name" in df.columns
    assert "caption_text" in df.columns


def test_split_train_val_shapes():
    df = load_captions(DUMMY_CAPTIONS_PATH)
    train_df, val_df = split_train_val(df, test_size=0.34, random_state=42)
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(train_df) + len(val_df) == len(df)


def test_split_train_val_no_image_leakage():
    df = load_captions(DUMMY_CAPTIONS_PATH)
    train_df, val_df = split_train_val(df, test_size=0.34, random_state=42)
    train_images = set(train_df["image_name"].unique())
    val_images = set(val_df["image_name"].unique())
    assert len(train_images.intersection(val_images)) == 0


def test_load_captions_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_captions("nonexistent_path/captions.txt")


# ---------- inference.py tests (using a generated dummy image, no real model needed) ----------


@pytest.fixture(scope="module")
def dummy_image_path(tmp_path_factory):
    """Generate a tiny in-memory test image, no real dataset required."""
    img_dir = tmp_path_factory.mktemp("dummy_images")
    img_path = os.path.join(img_dir, "dummy.jpg")
    image = Image.new("RGB", (64, 64), color=(120, 180, 90))
    image.save(img_path)
    return str(img_path)


def test_inference_module_importable():
    """Verifies inference.py has no import/syntax errors and exposes generate_caption."""
    from inference import generate_caption

    assert callable(generate_caption)


def test_generate_caption_raises_on_missing_image():
    """Edge case: generate_caption should raise cleanly on a nonexistent path,
    without needing a loaded model (fails at Image.open before model is used)."""
    from inference import generate_caption

    with pytest.raises(FileNotFoundError):
        generate_caption(None, None, "nonexistent_image.jpg", "cpu")


def test_dummy_image_fixture_is_valid_image(dummy_image_path):
    """Confirms our test fixture itself is a valid, openable image."""
    image = Image.open(dummy_image_path)
    assert image.size == (64, 64)
