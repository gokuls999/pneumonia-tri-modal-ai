# scripts/infer_single.py
import argparse
import json
from pathlib import Path
import uuid

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.training.tri_modal_model import TriModalPneumoniaNet
from src.explainability.gradcam_image import make_gradcam_heatmap, overlay_heatmap_on_image

# --------------------------------------------------
# Paths / config
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # project root (pneumonia-ai)
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


def load_img_as_tensor(path: Path, size: int = 224):
    """
    Read grayscale image -> (1,1,H,W) tensor + original np.uint8 image.
    """
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"Could not read image: {path}")
    orig = arr.copy()
    img = cv2.resize(arr, (size, size)).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1,H,W)
    img = np.expand_dims(img, axis=0)  # (1,1,H,W)
    tensor = torch.tensor(img, dtype=torch.float32)
    return tensor, orig


def heatmap_to_mask(heatmap, thr=0.5):
    """Create binary mask from 0–1 heatmap."""
    m = (heatmap >= thr).astype(np.uint8) * 255
    return m


def run_single_inference(xray_path, ct_path, text):
    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = TriModalPneumoniaNet(text_model_name="bert-base-uncased").to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE), strict=False)
    model.eval()

    # --------------------------------------------------
    # Preprocess inputs
    # --------------------------------------------------
    xray_tensor, x_orig = load_img_as_tensor(xray_path)
    xray_tensor = xray_tensor.to(DEVICE)

    ct_tensor = None
    ct_orig = None
    if ct_path is not None:
        ct_tensor, ct_orig = load_img_as_tensor(ct_path)
        ct_tensor = ct_tensor.to(DEVICE)
    else:
        # dummy CT (all zeros) if not provided
        ct_tensor = torch.zeros_like(xray_tensor).to(DEVICE)

    enc = TOKENIZER(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------
    with torch.no_grad():
        logits = model(xray_tensor, ct_tensor, input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        conf = float(probs[0, pred_idx].item())
        pred_label = "PNEUMONIA" if pred_idx == 1 else "NORMAL"

    # --------------------------------------------------
    # Grad-CAM X-ray
    # --------------------------------------------------
    x_grad_path = ""
    x_seg_path = ""
    try:
        x_heat = make_gradcam_heatmap(
            model=model,
            xray_tensor=xray_tensor,
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

        # overlay heatmap
        overlay_heatmap_on_image(x_orig, x_heat, GRADCAM_DIR / x_name)
        x_grad_path = str(GRADCAM_DIR / x_name)

        # segmentation mask
        h, w = x_orig.shape
        heat_resized = cv2.resize(x_heat, (w, h))
        x_mask = heatmap_to_mask(heat_resized, thr=0.4)
        cv2.imwrite(str(SEG_DIR / x_name), x_mask)
        x_seg_path = str(SEG_DIR / x_name)

    except Exception as e:
        print("X-ray Grad-CAM failed:", e)

    # --------------------------------------------------
    # Grad-CAM CT (if CT provided)
    # --------------------------------------------------
    ct_grad_path = ""
    ct_seg_path = ""
    if ct_orig is not None:
        try:
            ct_heat = make_gradcam_heatmap(
                model=model,
                xray_tensor=None,
                ct_tensor=ct_tensor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                device=DEVICE,
                target_class=pred_idx,
                backbone=model.ct_backbone_for_cam,
                branch="ct",
            )
            ct_heat = np.clip(ct_heat.astype(np.float32), 0.0, 1.0)
            c_name = f"{uuid.uuid4().hex}.png"

            overlay_heatmap_on_image(ct_orig, ct_heat, CT_GRADCAM_DIR / c_name)
            ct_grad_path = str(CT_GRADCAM_DIR / c_name)

            h2, w2 = ct_orig.shape
            ct_resized = cv2.resize(ct_heat, (w2, h2))
            ct_mask = heatmap_to_mask(ct_resized, thr=0.4)
            cv2.imwrite(str(CT_SEG_DIR / c_name), ct_mask)
            ct_seg_path = str(CT_SEG_DIR / c_name)

        except Exception as e:
            print("CT Grad-CAM failed:", e)

    # --------------------------------------------------
    # Build JSON-style result
    # --------------------------------------------------
    result = {
        "xray_path": str(xray_path),
        "ct_path": str(ct_path) if ct_path is not None else None,
        "clinical_text": text,
        "predicted_label": pred_label,
        "confidence": conf,
        "xray_gradcam": x_grad_path,
        "xray_segmentation": x_seg_path,
        "ct_gradcam": ct_grad_path,
        "ct_segmentation": ct_seg_path,
        "used_ct": ct_path is not None,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run tri-modal inference on a single case.")
    parser.add_argument("--xray", required=True, help="Path to X-ray image")
    parser.add_argument("--ct", required=False, help="Path to CT image (optional)")
    parser.add_argument("--text", required=True, help="Clinical text")

    args = parser.parse_args()

    xray_path = Path(args.xray)
    ct_path = Path(args.ct) if args.ct else None

    res = run_single_inference(xray_path, ct_path, args.text)
    print(json.dumps(res, indent=4))


if __name__ == "__main__":
    main()
