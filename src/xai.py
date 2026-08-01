import os
import numpy as np
import torch
from PIL import Image

def compute_vit_attention_map(model, processor, image_path, device):
    """CLS-token attention map (ViT self-attention, from Week 3 Day 2)."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        gen_out = model.generate(**inputs, num_beams=3, max_new_tokens=25)
    caption = processor.decode(gen_out[0], skip_special_tokens=True)

    full_inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        forward_out = model(
            pixel_values=full_inputs["pixel_values"],
            input_ids=full_inputs["input_ids"],
            output_attentions=True,
            return_dict=True
        )

    attn = forward_out.attentions[-1][0].mean(dim=0)
    cls_attn = attn[0, 1:].detach().cpu().numpy()
    grid_size = int(np.sqrt(cls_attn.shape[0]))
    heatmap = cls_attn.reshape(grid_size, grid_size)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    return image, heatmap, caption

if __name__ == "__main__":
    from model import load_baseline_model

    model_dir = os.environ.get("MODEL_DIR", "/models/week2_baseline_blip")
    image_path = os.environ.get("IMAGE_PATH", "/data/sample.jpg")

    model, processor, device = load_baseline_model(model_dir)
    _, heatmap, caption = compute_vit_attention_map(model, processor, image_path, device)
    print(f"Caption: {caption}")
    print(f"Heatmap shape: {heatmap.shape}")