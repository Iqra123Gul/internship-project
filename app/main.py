import os
import sys
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import load_baseline_model  # noqa: E402
from inference import generate_caption  # noqa: E402
from xai import compute_vit_attention_map  # noqa: E402

app = FastAPI(title="Flickr8k BLIP Captioning API")

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "models", "week2_baseline_blip"),
)
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "uploads")
OVERLAY_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "overlays")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

# Load model once at startup, reused across requests
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


@app.get("/")
def root():
    return {"status": "ok", "message": "Flickr8k BLIP Captioning API is running"}


@app.post("/caption")
async def caption_image(file: UploadFile = File(...), question: str = Form(None)):
    # Save uploaded image
    file_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")
    with open(image_path, "wb") as f:
        f.write(await file.read())

    # Generate caption
    caption = generate_caption(model, processor, image_path, device)

    # Compute Grad-CAM-style (ViT attention) overlay
    image, heatmap, _ = compute_vit_attention_map(model, processor, image_path, device)
    overlay = overlay_heatmap(image, heatmap)

    overlay_filename = f"{file_id}_overlay.png"
    overlay_path = os.path.join(OVERLAY_DIR, overlay_filename)
    Image.fromarray(overlay).save(overlay_path)

    response = {"caption": caption, "gradcam_overlay_path": overlay_path}

    if question:
        # Placeholder for VQA track — not implemented in this baseline model
        response["question"] = question
        response["note"] = (
            "VQA not supported by this baseline model; caption-only response returned."
        )

    return JSONResponse(content=response)
