import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_captions, split_train_val
from model import load_baseline_model
from inference import generate_caption

CAPTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Flickr8k", "captions.txt")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "Flickr8k", "images")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "week2_baseline_blip")


# ---------- data_loader.py tests ----------

def test_load_captions_returns_dataframe():
    df = load_captions(CAPTIONS_PATH)
    assert df is not None
    assert len(df) > 0
    assert "image_name" in df.columns
    assert "caption_text" in df.columns


def test_split_train_val_shapes():
    df = load_captions(CAPTIONS_PATH)
    train_df, val_df = split_train_val(df, test_size=0.1, random_state=42)
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(train_df) + len(val_df) == len(df)


def test_split_train_val_no_image_leakage():
    df = load_captions(CAPTIONS_PATH)
    train_df, val_df = split_train_val(df, test_size=0.1, random_state=42)
    train_images = set(train_df["image_name"].unique())
    val_images = set(val_df["image_name"].unique())
    assert len(train_images.intersection(val_images)) == 0


def test_load_captions_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_captions("nonexistent_path/captions.txt")


# ---------- inference.py tests ----------

@pytest.fixture(scope="module")
def loaded_model():
    model, processor, device = load_baseline_model(MODEL_DIR)
    return model, processor, device


def test_generate_caption_returns_string(loaded_model):
    model, processor, device = loaded_model
    sample_images = os.listdir(IMAGES_DIR)
    image_path = os.path.join(IMAGES_DIR, sample_images[0])
    caption = generate_caption(model, processor, image_path, device)
    assert isinstance(caption, str)
    assert len(caption) > 0


def test_generate_caption_nonempty_words(loaded_model):
    model, processor, device = loaded_model
    sample_images = os.listdir(IMAGES_DIR)
    image_path = os.path.join(IMAGES_DIR, sample_images[1])
    caption = generate_caption(model, processor, image_path, device)
    words = caption.split()
    assert len(words) > 0


def test_generate_caption_invalid_path_raises(loaded_model):
    model, processor, device = loaded_model
    with pytest.raises(FileNotFoundError):
        generate_caption(model, processor, "nonexistent_image.jpg", device)