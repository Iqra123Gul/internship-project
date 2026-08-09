# Flickr8k BLIP Image Captioning API

## Project Overview

This project develops an image captioning system using a fine-tuned
BLIP (Bootstrapping Language-Image Pre-training) model trained for
the Flickr8k image-captioning task.

The project covers the complete machine learning pipeline:

- Flickr8k dataset exploration
- Image and caption preprocessing
- BLIP model preparation
- Model registration and Hugging Face Hub upload
- FastAPI inference API
- Docker containerization
- Public deployment on Railway
- End-to-end testing with unseen images

---

## Architecture

```text
                    User
                     |
                     v
              Upload Image
                     |
                     v
        Railway Public Deployment
                     |
                     v
              FastAPI API
             POST /caption
                     |
                     v
             Image Preprocessing
                     |
                     v
             BLIP Processor
                     |
                     v
          Fine-tuned BLIP Model
                     |
                     v
            Caption Generation
                     |
                     v
              JSON Response
                     |
                     v
             Generated Caption