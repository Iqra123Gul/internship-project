import os
import sys
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import load_baseline_model
from inference import generate_caption

app = FastAPI(title="Flickr8k BLIP Captioning API")

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "IqraGulAI/flickr8k-blip-baseline",
)

UPLOAD_DIR = os.path.join(
    os.path.dirname(__file__), "uploads"
)

OVERLAY_DIR = os.path.join(
    os.path.dirname(__file__), "overlays"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

# Load model once at startup
model, processor, device = load_baseline_model(MODEL_DIR)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Flickr8k BLIP Captioning API is running"
    }


@app.post("/caption")
async def caption_image(
    file: UploadFile = File(...),
    question: str = Form(None)
):
    # Save uploaded image
    file_id = str(uuid.uuid4())
    image_path = os.path.join(
        UPLOAD_DIR,
        f"{file_id}.jpg"
    )

    with open(image_path, "wb") as f:
        f.write(await file.read())

    # Generate caption
    caption = generate_caption(
        model,
        processor,
        image_path,
        device,
        num_beams=1,
        max_new_tokens=15,
    )

    response = {
        "caption": caption
    }

    if question:
        response["question"] = question
        response["note"] = (
            "VQA not supported by this baseline model; "
            "caption-only response returned."
        )

    return JSONResponse(content=response)