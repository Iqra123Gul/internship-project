import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_captions(captions_path):
    """Load Flickr8k captions.txt (pipe-separated)."""
    df = pd.read_csv(captions_path, sep="|")
    return df


def split_train_val(captions_df, test_size=0.1, random_state=42):
    """Split by unique image to avoid leakage."""
    unique_images = captions_df["image_name"].unique()
    train_images, val_images = train_test_split(
        unique_images, test_size=test_size, random_state=random_state
    )
    train_df = captions_df[captions_df["image_name"].isin(train_images)].reset_index(
        drop=True
    )
    val_df = captions_df[captions_df["image_name"].isin(val_images)].reset_index(
        drop=True
    )
    return train_df, val_df


if __name__ == "__main__":
    captions_path = os.environ.get("CAPTIONS_PATH", "/data/Flickr8k/captions.txt")
    df = load_captions(captions_path)
    train_df, val_df = split_train_val(df)
    print(f"Loaded {len(df)} rows | Train: {len(train_df)} | Val: {len(val_df)}")
