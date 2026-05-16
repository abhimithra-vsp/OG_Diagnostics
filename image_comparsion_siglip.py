"""
Image comparison pipeline — SigLIP cosine + ORB/SIFT sub-image detection.

Pipeline:
  load → strip_solid_border → split_panels
       → SigLIP image embedding per panel (computed ONCE)
       → cosine similarity vs each target embedding
       → if not a global match, fall back to ORB+RANSAC sub-image search
         in both directions; if ORB can't classify, retry with SIFT
       → emit one of three public results

Public result strings (returned in `result` field):
  "match"
        Global SigLIP cosine ≥ THRESHOLD — same image up to recompression.
  "Given Image Contains Target Image"
        Sub-image relationship confirmed by ORB or SIFT — either strong
        containment, or strong inlier-density with meaningful overlap,
        or partial overlap with high inlier count. Covers cases like
        target image embedded inside a Canva graphic, possibly with text
        overlay, color filter, or cropping.
  "not match"
        Neither global SigLIP cosine nor feature-based sub-image search
        could establish a relationship.

Internal containment tier (containment["tier"], kept for QA debugging):
  "CONTAINED"  — strong containment (in_bounds ≥ STRICT or dense override)
  "DERIVED"    — partial overlap with high inlier count
"""

import time
import warnings
import numpy as np
import requests
import cv2
import torch
import os

from PIL import Image
from io import BytesIO

# HF env vars must be set BEFORE importing transformers
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from transformers import AutoImageProcessor, AutoModel

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CUDA BACKEND CONFIG
# =============================================================================
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


# =============================================================================
# CONFIG
# =============================================================================
MODEL_ID = "google/siglip-so400m-patch14-384"
# Drop-in upgrade once you're ready: "google/siglip2-so400m-patch14-384"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cosine threshold — SigLIP cosines run higher and sharper than CLIP.
THRESHOLD = 0.85

# Public result strings — exactly three statuses surfaced to callers.
# Internally the containment dict still carries a finer "tier" field
# ("CONTAINED" vs "DERIVED") for QA debugging.
RESULT_MATCH     = "match"
RESULT_CONTAINS  = "Given Image Contains Target Image"
RESULT_NOT_MATCH = "not match"

# Border stripping
BORDER_TOLERANCE   = 10
MIN_FOREGROUND_DIM = 80

# Collage detection / panel splitting
WHITE_ROW_THRESHOLD      = 245
WHITE_COL_THRESHOLD      = 245
DARK_SEPARATOR_THRESHOLD = 10
MIN_GAP_PX               = 8
MIN_PANEL_AREA_FRAC      = 0.04
MIN_PANEL_DIM            = 60

# ORB / sub-image detection
ORB_NFEATURES        = 3000
ORB_RATIO            = 0.75
ORB_MIN_MATCHES      = 10     # raw Lowe-ratio matches needed to attempt RANSAC
ORB_MIN_INLIERS      = 15     # post-RANSAC inliers required for ORB tier
STRICT_IN_BOUNDS     = 0.85   # CONTAINED tier
PARTIAL_IN_BOUNDS    = 0.50   # DERIVED tier
DERIVED_MIN_INLIERS  = 25     # extra inlier floor for DERIVED (partial overlap)
ORB_GATE_SIM         = 0.25   # skip ORB/SIFT on panels with SigLIP cosine below this

# SIFT fallback — invoked when ORB fails to classify. SIFT's gradient
# descriptors survive color filters, brightness shifts, and moderate
# stylization that destroy ORB's BRIEF intensity descriptors. Costs
# ~3-5× ORB but only runs when ORB couldn't classify.
SIFT_NFEATURES       = 3000
SIFT_RATIO           = 0.75
SIFT_MIN_INLIERS     = 12     # SIFT can use a lower bar; its inliers are higher quality

# Scale normalization — resize haystack & needle so longer edge is at most
# this many pixels before feature extraction. Closes the 1080-Canva vs
# 2000+px-raw-asset gap that pushes ORB's scale pyramid to its edge.
NORMALIZE_LONG_EDGE  = 1500

# Inlier-density override: when the geometric match is overwhelmingly strong
# (lots of RANSAC inliers) and there's meaningful overlap, treat as CONTAINED
# even if in_bounds is below STRICT_IN_BOUNDS. This catches "raw asset reused
# as full-bleed background under text/overlay" cases (e.g. Canva graphics
# where the photo fills the canvas but the bottom 35% is a gold gradient
# with a headline — see Run 2 (clinic photo) in the test set).
DENSE_OVERRIDE_INLIERS   = 100
DENSE_OVERRIDE_IN_BOUNDS = 0.50


# =============================================================================
# LOAD SIGLIP (module-level, once)
# =============================================================================
print(f"[INFO] Loading SigLIP on {DEVICE}: {MODEL_ID}")
processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
print("[INFO] SigLIP ready")


# =============================================================================
# IMAGE LOADING
# =============================================================================
def load_image(source):
    """Load to RGB numpy array. Caller must guarantee ndarray inputs are RGB."""
    if isinstance(source, np.ndarray):
        return source

    if isinstance(source, str) and source.startswith("http"):
        resp = requests.get(
            source,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(source).convert("RGB")

    return np.array(img)


# =============================================================================
# BORDER STRIP
# =============================================================================
def strip_solid_border(image, tolerance=BORDER_TOLERANCE):
    """Remove a uniform-color border (any color) around the image."""
    h, w = image.shape[:2]

    corners = np.array([
        image[0,  0],
        image[0, -1],
        image[-1, 0],
        image[-1, -1],
    ])
    border_color = np.median(corners, axis=0).astype(int)

    diff = np.abs(image.astype(int) - border_color).max(axis=2)
    content_mask = diff > tolerance

    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    if not rows.any() or not cols.any():
        return image

    r1 = int(np.argmax(rows))
    r2 = int(len(rows) - np.argmax(rows[::-1]) - 1)
    c1 = int(np.argmax(cols))
    c2 = int(len(cols) - np.argmax(cols[::-1]) - 1)

    stripped = image[r1:r2 + 1, c1:c2 + 1]
    sh, sw = stripped.shape[:2]

    if sw < MIN_FOREGROUND_DIM or sh < MIN_FOREGROUND_DIM:
        print("[STRIP] Foreground too small — returning original")
        return image

    return stripped


# =============================================================================
# COLLAGE / PANEL SPLITTING
# =============================================================================
def _find_separator_gaps(signal, threshold, min_gap):
    """Bright separators: stretches where signal >= threshold for >= min_gap pixels."""
    gaps, start = [], None
    for i, val in enumerate(signal):
        if val >= threshold and start is None:
            start = i
        elif val < threshold and start is not None:
            if i - start >= min_gap:
                gaps.append((start, i - 1))
            start = None
    if start is not None and len(signal) - start >= min_gap:
        gaps.append((start, len(signal) - 1))
    return gaps


def _find_dark_separator_gaps(signal, threshold, min_gap):
    """Dark separators: stretches where signal <= threshold for >= min_gap pixels."""
    gaps, start = [], None
    for i, val in enumerate(signal):
        if val <= threshold and start is None:
            start = i
        elif val > threshold and start is not None:
            if i - start >= min_gap:
                gaps.append((start, i - 1))
            start = None
    if start is not None and len(signal) - start >= min_gap:
        gaps.append((start, len(signal) - 1))
    return gaps


def _find_any_separator_gaps(signal, min_gap):
    """Try bright and dark separators; return whichever found more gaps."""
    bright = _find_separator_gaps(signal, WHITE_ROW_THRESHOLD, min_gap)
    dark   = _find_dark_separator_gaps(signal, DARK_SEPARATOR_THRESHOLD, min_gap)
    return bright if len(bright) >= len(dark) else dark


def split_panels(image):
    """Split a collage into panels; fall back to [image] if no valid panels."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)

    margin_h = max(1, h // 20)
    margin_w = max(1, w // 20)

    h_gaps_rel = _find_any_separator_gaps(row_mean[margin_h: h - margin_h], MIN_GAP_PX)
    v_gaps_rel = _find_any_separator_gaps(col_mean[margin_w: w - margin_w], MIN_GAP_PX)

    h_gaps = [(s + margin_h, e + margin_h) for s, e in h_gaps_rel]
    v_gaps = [(s + margin_w, e + margin_w) for s, e in v_gaps_rel]

    def make_bands(gaps, total):
        edges = [0]
        for gs, ge in gaps:
            edges.append(gs)
            edges.append(ge + 1)
        edges.append(total)
        bands = []
        for i in range(0, len(edges) - 1, 2):
            s, e = edges[i], edges[i + 1]
            if e > s:
                bands.append((s, e))
        return bands

    row_bands = make_bands(h_gaps, h)
    col_bands = make_bands(v_gaps, w)

    print(f"[SPLIT] row_bands={row_bands}  col_bands={col_bands}")

    total_area = h * w
    panels = []

    for rb_s, rb_e in row_bands:
        for cb_s, cb_e in col_bands:
            cell = image[rb_s:rb_e, cb_s:cb_e]
            ch, cw = cell.shape[:2]

            if cw < MIN_PANEL_DIM or ch < MIN_PANEL_DIM:
                continue
            if (ch * cw) / total_area < MIN_PANEL_AREA_FRAC:
                continue

            cell_mean = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY).mean()
            if cell_mean > 250 or cell_mean < 15:
                continue

            panels.append(cell)

    if not panels:
        print("[SPLIT] No valid panels found — using full image")
        return [image]

    print(f"[SPLIT] Extracted {len(panels)} panel(s)")
    return panels


# =============================================================================
# SIGLIP EMBEDDING
# =============================================================================
@torch.no_grad()
def get_siglip_embedding(image_np):
    """Return L2-normalized SigLIP image embedding for a numpy RGB image."""
    pil = Image.fromarray(image_np)
    inputs = processor(images=pil, return_tensors="pt").to(DEVICE)

    out = model.get_image_features(**inputs)

    # Unwrap if we got a ModelOutput instead of a raw tensor
    if not torch.is_tensor(out):
        if hasattr(out, "image_embeds") and out.image_embeds is not None:
            emb = out.image_embeds
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            emb = out.last_hidden_state.mean(dim=1)
        else:
            raise RuntimeError(
                f"Unexpected SigLIP output type {type(out)} with no usable embedding field"
            )
    else:
        emb = out

    return emb / emb.norm(dim=-1, keepdim=True)


def cosine(a, b):
    return torch.cosine_similarity(a, b).item()


# =============================================================================
# ORB / SIFT SUB-IMAGE DETECTION
# =============================================================================
def _quad_in_bounds_ratio(proj, w, h):
    """Fraction of the projected quad's area that lies within (0,0,w,h).
    Returns 0.0 if the quad is degenerate or non-convex (safe default)."""
    quad = proj.reshape(-1, 2).astype(np.float32)

    if not cv2.isContourConvex(quad):
        return 0.0

    full_area = cv2.contourArea(quad)
    if full_area <= 0:
        return 0.0

    img_rect = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    try:
        inter, _ = cv2.intersectConvexConvex(quad, img_rect)
    except cv2.error:
        return 0.0

    return float(inter) / float(full_area)


def _scale_if_larger(img, long_edge):
    """Downscale img so its longer edge is at most `long_edge`. Returns
    (scaled_img, scale_factor) where scaled_img == img and scale==1.0 if
    no downscaling was needed."""
    h, w = img.shape[:2]
    s = long_edge / max(h, w)
    if s >= 1.0:
        return img, 1.0
    return cv2.resize(img, (int(w * s), int(h * s)),
                      interpolation=cv2.INTER_AREA), s


def find_subimage(haystack, needle, method="orb",
                  min_matches=None,
                  min_inliers=None,
                  ratio=None,
                  nfeatures=None,
                  long_edge=NORMALIZE_LONG_EDGE):
    """
    Detect whether `needle` appears as a region within `haystack`, using
    ORB or SIFT keypoints.

    Both images are scale-normalized so that the longer edge of each is
    at most `long_edge` pixels before feature extraction. This closes the
    scale gap between 1080px Canva exports and 2000+px raw assets, which
    pushes ORB's scale pyramid to its edge. Projected bboxes are unscaled
    back to original haystack coordinates.

    Returns a dict:
      {
        "matched":   bool,
        "method":    "orb" | "sift",
        "inliers":   int,
        "in_bounds": float (0.0–1.0),
        "bbox":      (x1,y1,x2,y2) or None,   # in ORIGINAL haystack coords
        "reason":    str | None,
      }
    """
    # Per-detector defaults
    if method == "sift":
        min_matches = min_matches or ORB_MIN_MATCHES
        min_inliers = min_inliers or SIFT_MIN_INLIERS
        ratio       = ratio       or SIFT_RATIO
        nfeatures   = nfeatures   or SIFT_NFEATURES
        norm        = cv2.NORM_L2
        try:
            detector = cv2.SIFT_create(nfeatures=nfeatures)
        except AttributeError:
            return {"matched": False, "method": method, "inliers": 0,
                    "in_bounds": 0.0, "bbox": None,
                    "reason": "SIFT not available — install opencv-contrib-python or upgrade OpenCV ≥4.4"}
    else:  # orb
        method = "orb"
        min_matches = min_matches or ORB_MIN_MATCHES
        min_inliers = min_inliers or ORB_MIN_INLIERS
        ratio       = ratio       or ORB_RATIO
        nfeatures   = nfeatures   or ORB_NFEATURES
        norm        = cv2.NORM_HAMMING
        detector    = cv2.ORB_create(nfeatures=nfeatures)

    def fail(reason, inliers=0):
        return {"matched": False, "method": method, "inliers": inliers,
                "in_bounds": 0.0, "bbox": None, "reason": reason}

    # Scale-normalize both images
    hay_s, hay_scale = _scale_if_larger(haystack, long_edge)
    ndl_s, ndl_scale = _scale_if_larger(needle,   long_edge)

    g_hay = cv2.cvtColor(hay_s, cv2.COLOR_RGB2GRAY)
    g_ndl = cv2.cvtColor(ndl_s, cv2.COLOR_RGB2GRAY)

    kp_n, des_n = detector.detectAndCompute(g_ndl, None)
    kp_h, des_h = detector.detectAndCompute(g_hay, None)

    if des_n is None or des_h is None or len(kp_n) < 4 or len(kp_h) < 4:
        return fail("too few keypoints")

    bf = cv2.BFMatcher(norm)
    raw = bf.knnMatch(des_n, des_h, k=2)

    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m1, m2 = pair
        if m1.distance < ratio * m2.distance:
            good.append(m1)

    if len(good) < min_matches:
        return fail(f"only {len(good)} Lowe-ratio matches (need {min_matches})")

    src = np.float32([kp_n[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_h[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return fail("homography failed")

    inliers = int(mask.sum())
    if inliers < min_inliers:
        return fail(f"only {inliers} RANSAC inliers (need {min_inliers})", inliers)

    # Project needle corners (in SCALED needle space) through H
    hn_s, wn_s = ndl_s.shape[:2]
    corners = np.float32([[0, 0], [wn_s, 0], [wn_s, hn_s], [0, hn_s]]).reshape(-1, 1, 2)
    proj_scaled = cv2.perspectiveTransform(corners, H)

    # in_bounds is a ratio, scale-invariant; compute against scaled haystack
    hh_s, wh_s = hay_s.shape[:2]
    in_bounds = _quad_in_bounds_ratio(proj_scaled, wh_s, hh_s)

    # Unscale the projected quad back to ORIGINAL haystack coordinates for bbox
    proj_orig = proj_scaled / hay_scale
    x1, y1 = proj_orig.min(axis=0).ravel()
    x2, y2 = proj_orig.max(axis=0).ravel()

    return {
        "matched":   True,
        "method":    method,
        "inliers":   inliers,
        "in_bounds": float(in_bounds),
        "bbox":      (int(x1), int(y1), int(x2), int(y2)),
        "reason":    None,
    }


# Backwards-compatible alias for callers that still reference the old name
def find_subimage_orb(haystack, needle, **kwargs):
    return find_subimage(haystack, needle, method="orb", **kwargs)


def _classify_orb_result(r):
    """Map a find_subimage result dict to a tier string or None.

    Tiers:
      CONTAINED — strong containment by either:
                    (a) in_bounds ≥ STRICT_IN_BOUNDS, OR
                    (b) inlier-density override: inliers ≥ DENSE_OVERRIDE_INLIERS
                        AND in_bounds ≥ DENSE_OVERRIDE_IN_BOUNDS
                        (raw asset reused as background under text overlay)
      DERIVED   — partial overlap (shared crop region)
      None      — match too weak to report
    """
    if not r["matched"]:
        return None
    if r["in_bounds"] >= STRICT_IN_BOUNDS:
        return "CONTAINED"
    if (r["inliers"]   >= DENSE_OVERRIDE_INLIERS and
        r["in_bounds"] >= DENSE_OVERRIDE_IN_BOUNDS):
        return "CONTAINED"
    if r["in_bounds"] >= PARTIAL_IN_BOUNDS and r["inliers"] >= DERIVED_MIN_INLIERS:
        return "DERIVED"
    return None


def _orb_candidate_is_better(new_cand, best_cand):
    """Prefer CONTAINED over DERIVED, then higher inlier count."""
    if best_cand is None:
        return True
    tier_rank = {"CONTAINED": 2, "DERIVED": 1}
    new_rank  = tier_rank[new_cand["tier"]]
    best_rank = tier_rank[best_cand["tier"]]
    if new_rank != best_rank:
        return new_rank > best_rank
    return new_cand["inliers"] > best_cand["inliers"]


# =============================================================================
# MAIN COMPARISON
# =============================================================================
def compare_images(source1, source2):
    """
    Compare a source image (possibly a collage) against one or more target
    images. Returns the overall best match plus full per-target breakdown
    in `all_results`.
    """
    try:
        total_start = time.time()
        print(f"\n[START] {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # ── source: load, strip border, split into panels ─────────────────────
        img1 = strip_solid_border(load_image(source1))
        panels = split_panels(img1)

        print(f"[INFO] Embedding {len(panels)} source panel(s)")
        panel_embs = [get_siglip_embedding(p) for p in panels]

        # ── normalize source2 to a list ───────────────────────────────────────
        if isinstance(source2, str):
            source2_list = [source2]
        elif isinstance(source2, list):
            source2_list = source2
        else:
            raise ValueError("source2 must be a string or list of strings")

        all_results = []

        for target_idx, target_source in enumerate(source2_list):
            print(f"\n[INFO] TARGET {target_idx + 1}/{len(source2_list)}: {target_source}")

            # JFIF skip — per-target, does not abort the batch
            if isinstance(target_source, str) and target_source.lower().endswith(".jfif"):
                print("[SKIP] JFIF format — manual check required")
                all_results.append({
                    "target_index": target_idx + 1,
                    "target_image": target_source,
                    "result":       "SKIPPED",
                    "reason":       "JFIF format — please check manually",
                })
                continue

            try:
                img2 = strip_solid_border(load_image(target_source))
                target_emb = get_siglip_embedding(img2)
            except Exception as e:
                print(f"[ERROR] Target load/embed failed: {e}")
                all_results.append({
                    "target_index": target_idx + 1,
                    "target_image": target_source,
                    "result":       "ERROR",
                    "reason":       str(e),
                })
                continue

            # ── SigLIP cosine, per panel ─────────────────────────────────────
            panel_sims = []
            best_similarity = -1.0
            best_panel = None
            for idx, panel_emb in enumerate(panel_embs):
                sim = cosine(panel_emb, target_emb)
                panel_sims.append(sim)
                print(f"[INFO] Panel {idx + 1} similarity: {sim:.4f}")
                if sim > best_similarity:
                    best_similarity = sim
                    best_panel = idx + 1

            result = RESULT_MATCH if best_similarity >= THRESHOLD else RESULT_NOT_MATCH
            method = "SigLIP cosine"
            containment = None

            # ── ORB sub-image search (only if not a global MATCH) ─────────────
            #
            # Strategy: try ORB first (fast, intensity-based descriptors).
            # If ORB cannot classify the pair for a given (panel, direction),
            # retry with SIFT (slower but gradient-based; survives color
            # filters, brightness shifts, and Canva-style stylization that
            # break ORB's BRIEF descriptors). Both detectors run with scale
            # normalization to close the 1080-Canva vs 2000+-raw gap.
            if result == RESULT_NOT_MATCH:
                best_cand = None

                for idx, panel in enumerate(panels):
                    # Skip detectors on panels SigLIP rated as wholly unrelated
                    if panel_sims[idx] < ORB_GATE_SIM:
                        print(f"[GATE] Panel {idx + 1}: skipped "
                              f"(SigLIP {panel_sims[idx]:.3f} < gate {ORB_GATE_SIM})")
                        continue

                    # Evaluate BOTH directions; keep the best across all panels
                    for direction_label, hay, nee in [
                        ("target_in_source", panel, img2),
                        ("source_in_target", img2, panel),
                    ]:
                        # Pass 1: ORB
                        r = find_subimage(hay, nee, method="orb")
                        tier = _classify_orb_result(r)
                        print(f"[ORB ] Panel {idx + 1} {direction_label}: "
                              f"matched={r['matched']} inliers={r['inliers']} "
                              f"in_bounds={r['in_bounds']:.3f} tier={tier}")

                        # Pass 2: SIFT fallback if ORB couldn't classify.
                        # Always try SIFT when ORB fails — the SigLIP gate
                        # already filtered wholly-unrelated panels.
                        if tier is None:
                            r_sift = find_subimage(hay, nee, method="sift")
                            tier_sift = _classify_orb_result(r_sift)
                            print(f"[SIFT] Panel {idx + 1} {direction_label}: "
                                  f"matched={r_sift['matched']} "
                                  f"inliers={r_sift['inliers']} "
                                  f"in_bounds={r_sift['in_bounds']:.3f} "
                                  f"tier={tier_sift}")
                            if tier_sift is not None:
                                r    = r_sift
                                tier = tier_sift

                        if tier is None:
                            continue

                        cand = {
                            "direction": direction_label,
                            "panel":     idx + 1,
                            "detector":  r["method"],      # "orb" or "sift"
                            "inliers":   r["inliers"],
                            "in_bounds": round(r["in_bounds"], 3),
                            "bbox":      r["bbox"],
                            "tier":      tier,
                        }
                        if _orb_candidate_is_better(cand, best_cand):
                            best_cand = cand

                if best_cand is not None:
                    containment = best_cand   # tier kept internally for QA
                    result = RESULT_CONTAINS  # CONTAINED ∪ DERIVED → one public status
                    method = f"SigLIP cosine + {best_cand['detector'].upper()}/RANSAC"

            # ── assemble per-target record ────────────────────────────────────
            record = {
                "target_index":  target_idx + 1,
                "target_image":  target_source,
                "matched_panel": (containment["panel"] if containment else best_panel),
                "similarity":    round(best_similarity, 4),
                "threshold":     THRESHOLD,
                "method":        method,
                "result":        result,
            }
            if containment:
                record["containment"] = containment
            all_results.append(record)

        elapsed = round(time.time() - total_start, 2)

        # ── pick overall best across non-skipped/non-error targets ────────────
        scored = [r for r in all_results if "similarity" in r]
        if not scored:
            return {
                "result":          "Matched but check manual review",
                "all_results":     all_results,
                "elapsed_seconds": elapsed,
            }

        # Rank by verdict tier first, then by similarity (SigLIP).
        # A "Given Image Contains Target Image" finding with low SigLIP
        # cosine still beats a plain "not match".
        tier_rank = {
            RESULT_MATCH:     2,
            RESULT_CONTAINS:  1,
            RESULT_NOT_MATCH: 0,
        }
        overall_best = max(
            scored,
            key=lambda r: (tier_rank.get(r["result"], 0), r["similarity"])
        )

        return {
            "similarity":      overall_best["similarity"],
            "threshold":       overall_best["threshold"],
            "method":          overall_best["method"],
            "matched_panel":   overall_best["matched_panel"],
            "best_target":     overall_best["target_index"],
            "result":          overall_best["result"],            
            "elapsed_seconds": elapsed,
        }

    except Exception as e:
        return {"error": str(e), "result": "ERROR"}


# =============================================================================
# ENTRYPOINT
# =============================================================================
if __name__ == "__main__":
    url1 = "https://dify-assets.s3.amazonaws.com/assets/a380ba4e-cefe-441f-9776-ed93fea64830/e3d1c38c-d21a-11f0-95e5-008cfaff2740/DAHIF3PHpoo_canva_1777332436179.jpeg"

    url2 = [
        "https://img1.wsimg.com/isteam/ip/a18b3dc5-6256-430b-84cc-2e052dd07a5f/Aragon-Headshot-Junie-Dubois-43.jpg"
    ]

    output = compare_images(url1, url2)

    print("\n=== FINAL RESULT ===")
    for k, v in output.items():
        if k == "all_results":
            print(f"  {k}:")
            for r in v:
                print(f"    - {r}")
        else:
            print(f"  {k:18s}: {v}")