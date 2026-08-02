\# Internship Project



This repository contains my internship tasks.



\## Folder Structure



\- data/

\- notebooks/

\- src/

\- tests/

## Day 3

Installed required libraries and practiced Git branching.



Author: Iqra Gul
## Running the FastAPI App with Docker

### Build the image
\`\`\`bash
docker build -f docker/app.Dockerfile -t flickr8k-api .
\`\`\`

### Run the container
The model weights are mounted as a volume (not baked into the image) to keep the image size small:

\`\`\`bash
docker run --rm -p 8000:8000 -v C:\Users\Lenovo\Internship\models\week2_baseline_blip:/app/models/week2_baseline_blip flickr8k-api
\`\`\`

Adjust the volume path on the left of the `:` to match your local model directory if different.

### Test the API
Once running, the API is available at `http://127.0.0.1:8000`.

**Via curl:**
\`\`\`bash
curl.exe -X POST "http://127.0.0.1:8000/caption" -F "file=@data/Flickr8k/images/YOUR_IMAGE.jpg"
\`\`\`

**Via browser (interactive Swagger UI):**
Visit `http://127.0.0.1:8000/docs`

### Response format
\`\`\`json
{
  "caption": "a man wearing a red jacket sits on a park bench.",
  "gradcam_overlay_path": "/app/app/overlays/<uuid>_overlay.png"
}
\`\`\`

### Notes on debugging environment issues
The Dockerfile installs `libgl1` and `libglib2.0-0` system libraries and uses
`opencv-python-headless` (instead of `opencv-python`) to avoid the common
`ImportError: libGL.so.1: cannot open shared object file` error that occurs
when running OpenCV inside a minimal Linux container without GUI libraries.
