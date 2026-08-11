import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_baseline_model(model_dir, device=None):
    """Load the fine-tuned BLIP baseline model + processor."""
    
    device = device or "cpu"
    
    print("Loading processor...", flush=True)
    processor = BlipProcessor.from_pretrained(model_dir)
    print("Processor loaded.", flush=True)

    print("Loading model...", flush=True)
    model = BlipForConditionalGeneration.from_pretrained(
        model_dir,
        low_cpu_mem_usage=True
    )
    
    model = model.to(device)
    model.eval()

    print("Model loaded successfully.", flush=True)
    return model, processor, device


if __name__ == "__main__":
    model_dir = os.environ.get(
        "MODEL_DIR",
        "/models/week2_baseline_blip"
    )

    model, processor, device = load_baseline_model(model_dir)
    print(f"Model loaded on {device}", flush=True)import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_baseline_model(model_dir, device=None):
    """Load the fine-tuned BLIP baseline model + processor."""
    
    device = device or "cpu"
    
    print("Loading processor...", flush=True)
    processor = BlipProcessor.from_pretrained(model_dir)
    print("Processor loaded.", flush=True)

    print("Loading model...", flush=True)
    model = BlipForConditionalGeneration.from_pretrained(
        model_dir,
        low_cpu_mem_usage=True
    )
    
    model = model.to(device)
    model.eval()

    print("Model loaded successfully.", flush=True)
    return model, processor, device


if __name__ == "__main__":
    model_dir = os.environ.get(
        "MODEL_DIR",
        "/models/week2_baseline_blip"
    )

    model, processor, device = load_baseline_model(model_dir)
    print(f"Model loaded on {device}", flush=True)