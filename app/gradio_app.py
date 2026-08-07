import os
import sys
import numpy as np
import cv2
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import load_baseline_model  # noqa: E402
from inference import generate_caption  # noqa: E402
from xai import compute_vit_attention_map  # noqa: E402

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "models", "week2_baseline_blip"),
)

model, processor, device = load_baseline_model(MODEL_DIR)


def overlay_heatmap(image, heatmap, alpha=0.5):
    image_np = np.array(image.resize((384, 384)))
    heatmap_resized = cv2.resize(heatmap, (384, 384))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_colored + (1 - alpha) * image_np).astype(np.uint8)
    return overlay


def process_image(input_image):
    if input_image is None:
        return None, "Please upload an image."

    temp_path = "temp_upload.jpg"
    input_image.save(temp_path)

    caption = generate_caption(model, processor, temp_path, device)
    image, heatmap, _ = compute_vit_attention_map(model, processor, temp_path, device)
    overlay = overlay_heatmap(image, heatmap)

    os.remove(temp_path)

    return overlay, caption


demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Image(type="numpy", label="Grad-CAM Overlay"),
        gr.Textbox(label="Generated Caption"),
    ],
    title="Flickr8k BLIP Captioning + Explainability Demo",
    description="Upload an image to generate a caption and see which regions the model focused on.",
)

if __name__ == "__main__":
    demo.launch()
