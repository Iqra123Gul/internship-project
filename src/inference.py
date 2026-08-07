import os
from PIL import Image
import torch


def generate_caption(
    model, processor, image_path, device, num_beams=3, max_new_tokens=25
):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, num_beams=num_beams, max_new_tokens=max_new_tokens
        )
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


if __name__ == "__main__":
    from model import load_baseline_model

    model_dir = os.environ.get("MODEL_DIR", "/models/week2_baseline_blip")
    image_path = os.environ.get("IMAGE_PATH", "/data/sample.jpg")

    model, processor, device = load_baseline_model(model_dir)
    caption = generate_caption(model, processor, image_path, device)
    print(f"Caption: {caption}")
