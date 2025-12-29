# src/inference/tri_infer_core.py

import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# project root
ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.append(str(ROOT))

from src.training.tri_modal_model import TriModalPneumoniaNet
from src.explainability.gradcam_image import make_gradcam_heatmap, overlay_heatmap_on_image


# -----------------------------
# PATHS / GLOBALS
# -----------------------------
CKPT = ROOT / "models" / "checkpoints" / "tri_modal_baseline.pth"

MEDIA_ROOT = ROOT / "media"
GRADCAM_DIR = MEDIA_ROOT / "gradcam"
SEG_DIR = MEDIA_ROOT / "segmentation"
CT_GRADCAM_DIR = MEDIA_ROOT / "ct_gradcam"
CT_SEG_DIR = MEDIA_ROOT / "ct_seg"

for d in (GRADCAM_DIR, SEG_DIR, CT_GRADCAM_DIR, CT_SEG_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")

_MODEL = None  # lazy-loaded global model


# -----------------------------
# HELPERS
# -----------------------------
def _load_model():
    """
    Lazy-load the tri-modal model once and reuse it for all inferences.
    """
    global _MODEL
    if _MODEL is None:
        model = TriModalPneumoniaNet(
            text_model_name="bert-base-uncased",
            num_classes=2,
            fused_hidden_size=512,
        ).to(DEVICE)
        # important: strict=False because ckpt does not have CAM copies
        state = torch.load(CKPT, map_location=DEVICE)
        _ = model.load_state_dict(state, strict=False)
        model.eval()
        _MODEL = model
    return _MODEL


def _load_img_as_tensor(path, size=224, force_channel=1):
    """
    Reads image as grayscale and returns:
      tensor: (1, C, H, W)
      orig:   original grayscale numpy array
    """
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"Could not read image: {path}")

    orig = arr.copy()
    img = cv2.resize(arr, (size, size)).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W)
    if force_channel == 3:
        img = np.repeat(img, 3, axis=0)  # (3, H, W)
    tensor = torch.tensor(np.expand_dims(img, 0), dtype=torch.float32)  # (1, C, H, W)
    return tensor, orig


def _heatmap_to_mask(heatmap, thr=0.4):
    """
    Convert 0–1 heatmap to 0/255 mask.
    """
    m = (heatmap >= thr).astype(np.uint8) * 255
    return m


# -----------------------------
# PUBLIC API
# -----------------------------
def run_tri_inference(xray_path, ct_path, clinical_text):
    """
    Main callable for Django / API.

    Args:
        xray_path (str or Path): path to chest x-ray image
        ct_path   (str or Path or None): path to CT slice (optional, but here we expect it)
        clinical_text (str): clinical notes text

    Returns:
        dict with:
          xray_path, ct_path, clinical_text,
          predicted_label, confidence,
          xray_gradcam, xray_segmentation,
          ct_gradcam, ct_segmentation,
          used_ct (bool)
    """
    model = _load_model()

    xray_path = Path(xray_path)
    ct_path = Path(ct_path) if ct_path is not None else None

    # ---- preprocess images ----
    x_t, x_orig = _load_img_as_tensor(xray_path, force_channel=1)
    if ct_path is not None:
        ct_t, ct_orig = _load_img_as_tensor(ct_path, force_channel=1)
        used_ct = True
    else:
        # dummy ct
        ct_t = torch.zeros_like(x_t)
        ct_orig = None
        used_ct = False

    # ---- preprocess text ----
    enc = TOKENIZER(
        clinical_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    x_t = x_t.to(DEVICE)
    ct_t = ct_t.to(DEVICE)

    # ---- forward pass ----
    with torch.no_grad():
        logits = model(x_t, ct_t, input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        conf = float(probs[0, pred_idx].item())
        pred_label = "PNEUMONIA" if pred_idx == 1 else "NORMAL"

    # ---- X-RAY GRAD-CAM ----
    x_grad_path = ""
    x_seg_path = ""
    try:
        from src.training.tri_modal_model import TriModalPneumoniaNet  # just to keep type hints local, no-op

        x_heat = make_gradcam_heatmap(
            model=model,
            xray_tensor=x_t,
            ct_tensor=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=DEVICE,
            target_class=pred_idx,
            backbone=model.xray_backbone_for_cam,
            branch="xray",
        )

        x_heat = np.clip(x_heat.astype(np.float32), 0.0, 1.0)
        x_name = f"{uuid.uuid4().hex}.png"

        x_grad_path = GRADCAM_DIR / x_name
        x_seg_path = SEG_DIR / x_name

        overlay_heatmap_on_image(x_orig, x_heat, x_grad_path)

        heat_resized = cv2.resize(x_heat, (x_orig.shape[1], x_orig.shape[0]))
        mask = _heatmap_to_mask(heat_resized, thr=0.4)
        cv2.imwrite(str(x_seg_path), mask)
    except Exception as e:
        print("[tri_infer_core] xray gradcam failed:", e)
        x_grad_path = ""
        x_seg_path = ""

    # ---- CT GRAD-CAM ----
    ct_grad_path = ""
    ct_seg_path = ""
    if used_ct:
        try:
            ct_heat = make_gradcam_heatmap(
                model=model,
                xray_tensor=None,
                ct_tensor=ct_t,
                input_ids=input_ids,
                attention_mask=attention_mask,
                device=DEVICE,
                target_class=pred_idx,
                backbone=model.ct_backbone_for_cam,
                branch="ct",
            )

            ct_heat = np.clip(ct_heat.astype(np.float32), 0.0, 1.0)
            ct_name = f"{uuid.uuid4().hex}.png"

            ct_grad_path = CT_GRADCAM_DIR / ct_name
            ct_seg_path = CT_SEG_DIR / ct_name

            overlay_heatmap_on_image(ct_orig, ct_heat, ct_grad_path)

            ct_resized = cv2.resize(ct_heat, (ct_orig.shape[1], ct_orig.shape[0]))
            ct_mask = _heatmap_to_mask(ct_resized, thr=0.4)
            cv2.imwrite(str(ct_seg_path), ct_mask)
        except Exception as e:
            print("[tri_infer_core] ct gradcam failed:", e)
            ct_grad_path = ""
            ct_seg_path = ""

    # ---- build response dict ----
    result = {
        "xray_path": str(xray_path),
        "ct_path": str(ct_path) if ct_path is not None else "",
        "clinical_text": clinical_text,
        "predicted_label": pred_label,
        "confidence": conf,
        "xray_gradcam": str(x_grad_path) if x_grad_path else "",
        "xray_segmentation": str(x_seg_path) if x_seg_path else "",
        "ct_gradcam": str(ct_grad_path) if ct_grad_path else "",
        "ct_segmentation": str(ct_seg_path) if ct_seg_path else "",
        "used_ct": bool(used_ct),
    }

    return result

# Backwards-compatible name used by Django API
def tri_infer_single(xray_path, ct_path, clinical_text):
    """
    Small wrapper so existing imports still work.
    """
    return run_tri_inference(xray_path, ct_path, clinical_text)
