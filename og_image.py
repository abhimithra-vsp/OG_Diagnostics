import time
import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# -----------------------------
# CONFIG
# -----------------------------
SAM_CHECKPOINT = "utils/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
THRESHOLD = 0.85   # adjust based on strictness

device = "cpu"

# -----------------------------
# LOAD SAM
# -----------------------------
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

# -----------------------------
# LOAD CLIP
# -----------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_fast=False
)

# -----------------------------
# LOAD IMAGE FROM URL
# -----------------------------
def load_image(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(img)

# -----------------------------
# EXTRACT MAIN OBJECT (SAM)
# -----------------------------
def extract_main_object(image, max_size=1024):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = Image.fromarray(image).resize((new_w, new_h), Image.LANCZOS)
        image = np.array(resized)

    masks = mask_generator.generate(image)

    if not masks:
        return image  # fallback

    largest_mask = max(masks, key=lambda x: x['area'])
    segmentation = largest_mask['segmentation']

    masked = image.copy()
    masked[~segmentation] = 0

    return masked

# -----------------------------
# GET EMBEDDING (CLIP)
# -----------------------------
def get_embedding(image):
    pil_img = Image.fromarray(image)
    inputs = clip_processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = vision_outputs.pooler_output if hasattr(vision_outputs, "pooler_output") else vision_outputs[1]
        emb = clip_model.visual_projection(pooled)

    return emb / emb.norm(dim=-1, keepdim=True)

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def compare_images(url1, url2):
    try:
        start = time.time()
        print(f"[START] Processing started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        t0 = time.time()
        img1 = load_image(url1)
        img2 = load_image(url2)
        print(f"[TIME] Image loading: {time.time() - t0:.2f}s")

        t0 = time.time()
        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)
        print(f"[TIME] CLIP embedding: {time.time() - t0:.2f}s")

        t0 = time.time()
        similarity = torch.cosine_similarity(emb1, emb2).item()
        result = "MATCH" if similarity >= THRESHOLD else "NOT MATCH"
        print(f"[TIME] Similarity computation: {time.time() - t0:.2f}s")

        total = time.time() - start
        print(f"[END] Processing finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[TOTAL] Elapsed time: {total:.2f}s")

        return {
            "similarity": round(similarity, 4),
            "result": result,
            "elapsed_seconds": round(total, 2)
        }

    except Exception as e:
        return {
            "error": str(e),
            "result": "ERROR"
        }

# -----------------------------
# USAGE
# -----------------------------
url1 = "https://dify-assets.s3.amazonaws.com/assets/312d8eb2-d779-4bed-adf1-6021602f229c/8c665b7b-8d17-11ed-82a1-7cd30ab1652a/DAHHfI7gMMw_canva_1776775944953.jpeg"
url2 = "https://img1.wsimg.com/isteam/ip/c1963d0f-8ea9-4be0-9c2e-565fe47cf11f/Red%20Cream%20Delicate%20Quirky%20Illustration%20Wedding.jpg"
script_start = time.time()
output = compare_images(url1, url2)
print(output)
print(f"[OVERALL] Script total time: {time.time() - script_start:.2f}s")