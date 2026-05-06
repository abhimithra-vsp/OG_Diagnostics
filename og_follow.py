import time
import torch
import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# =========================================================
# CONFIG
# =========================================================

CSV_INPUT = "extracted_instruction_image_only.csv"
CSV_OUTPUT = "og_image_comparison_output.csv"

SAM_CHECKPOINT = "utils/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

THRESHOLD = 0.85
DEVICE = "cpu"

# =========================================================
# LOAD SAM
# =========================================================

print("[INFO] Loading SAM model...")

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(DEVICE)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crop_n_layers=0
)

# =========================================================
# LOAD CLIP
# =========================================================

print("[INFO] Loading CLIP model...")

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(DEVICE)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_fast=False
)

# =========================================================
# LOAD IMAGE
# =========================================================

def load_image(url):
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(img)

# =========================================================
# EXTRACT MAIN OBJECT USING SAM
# =========================================================

def extract_main_object(image, max_size=1024):

    h, w = image.shape[:2]

    if max(h, w) > max_size:
        scale = max_size / max(h, w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        resized = Image.fromarray(image).resize(
            (new_w, new_h),
            Image.LANCZOS
        )

        image = np.array(resized)

    masks = mask_generator.generate(image)

    if not masks:
        return image

    largest_mask = max(masks, key=lambda x: x["area"])

    segmentation = largest_mask["segmentation"]

    masked = image.copy()
    masked[~segmentation] = 0

    return masked

# =========================================================
# CLIP EMBEDDING
# =========================================================

def get_embedding(image):

    pil_img = Image.fromarray(image)

    inputs = clip_processor(
        images=pil_img,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        vision_outputs = clip_model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        pooled = (
            vision_outputs.pooler_output
            if hasattr(vision_outputs, "pooler_output")
            else vision_outputs[1]
        )

        emb = clip_model.visual_projection(pooled)

    emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb

# =========================================================
# IMAGE COMPARISON
# =========================================================

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

# =========================================================
# PROCESS CSV
# =========================================================

def process_csv():

    overall_start = time.time()

    print(f"[INFO] Reading CSV: {CSV_INPUT}")

    df = pd.read_csv(CSV_INPUT)

    output_rows = []

    total_rows = len(df)

    print(f"[INFO] Total rows found: {total_rows}")

    for index, row in df.iterrows():

        try:

            order_id = row.get("Iris ID", "")

            url1 = row.get("Complete Toggle Data - imageUrlP1", "")
            url2 = row.get("Edit Instruction (Input)", "")

            print("\n" + "=" * 80)
            print(f"[{index + 1}/{total_rows}] Processing: {order_id}")

            # Skip empty URLs
            if pd.isna(url1) or pd.isna(url2):
                print("[SKIP] Missing image URL")

                output_rows.append({
                    "Order ID": order_id,
                    "our using image": url1,
                    "customer image": url2,
                    "status": "MISSING URL",
                    "time": "0s"
                })

                continue

            print(f"[URL1] {url1}")
            print(f"[URL2] {url2}")

            result = compare_images(url1, url2)

            print(f"[RESULT] {result['result']}")
            print(f"[SIMILARITY] {result['similarity']}")
            print(f"[TIME] {result['elapsed_seconds']}s")

            output_rows.append({
                "Order ID": order_id,
                "our using image": url1,
                "customer image": url2,
                "status": result["result"],
                "time": f"{result['elapsed_seconds']}s"
            })

        except Exception as e:

            print(f"[ERROR] {str(e)}")

            output_rows.append({
                "Order ID": order_id if 'order_id' in locals() else "",
                "our using image": url1 if 'url1' in locals() else "",
                "customer image": url2 if 'url2' in locals() else "",
                "status": f"ERROR: {str(e)}",
                "time": "0s"
            })

    # =====================================================
    # SAVE OUTPUT
    # =====================================================

    output_df = pd.DataFrame(output_rows)

    output_df.to_csv(CSV_OUTPUT, index=False)

    total_elapsed = round(time.time() - overall_start, 2)

    print("\n" + "=" * 80)
    print(f"[DONE] CSV saved: {CSV_OUTPUT}")
    print(f"[TOTAL TIME] {total_elapsed}s")
    print("=" * 80)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    process_csv()