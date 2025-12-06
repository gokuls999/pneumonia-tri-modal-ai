# src/inference/eval_tri_inference.py
import csv
import json
import uuid
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import AutoTokenizer

# small wrapper that exposes `image_backbone` while forwarding other attributes
import types
import torch.nn as nn

class ModelWithBackbone(nn.Module):
    def __init__(self, orig_model, backbone):
        super().__init__()
        self._orig = orig_model
        # expose the backbone attribute the gradcam helper expects
        self.image_backbone = backbone

    def __getattr__(self, name):
        # forward attribute access to the original model for everything else
        if name in ("_orig", "image_backbone"):
            return object.__getattribute__(self, name)
        return getattr(self._orig, name)


# project root
ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.append(str(ROOT))

from src.training.tri_modal_model import TriModalPneumoniaNet
from src.explainability.gradcam_image import make_gradcam_heatmap, overlay_heatmap_on_image

# paths / config
CSV_VAL = ROOT / "data" / "processed" / "tri_dataset_v2_val.csv"
CKPT = ROOT / "models" / "checkpoints" / "tri_modal_baseline.pth"
OUT_CSV = ROOT / "data" / "processed" / "tri_val_outputs.csv"
MEDIA_ROOT = ROOT / "media"
GRADCAM_DIR = MEDIA_ROOT / "gradcam"
SEG_DIR = MEDIA_ROOT / "segmentation"
CT_GRADCAM_DIR = MEDIA_ROOT / "ct_gradcam"
CT_SEG_DIR = MEDIA_ROOT / "ct_seg"
for d in (GRADCAM_DIR, SEG_DIR, CT_GRADCAM_DIR, CT_SEG_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")

# helper: read image file -> model tensor + orig image (grayscale)
def load_img_as_tensor(path, size=224, force_channel=1):
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

# small thresholder -> segmentation mask from heatmap (0..1)
def heatmap_to_mask(heatmap, thr=0.5):
    m = (heatmap >= thr).astype(np.uint8) * 255
    return m

def main():
    # model
    model = TriModalPneumoniaNet(text_model_name="bert-base-uncased").to(DEVICE)

    state = torch.load(CKPT, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)

    model.eval()

    rows_out = []
    with open(CSV_VAL, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in tqdm(rows, desc="Eval val"):
        xray_p = Path(row["xray_path"])
        ct_p = Path(row["ct_path"])
        text = row["clinical_text"]
        label = row.get("label", "")

        try:
            x_t, x_orig = load_img_as_tensor(xray_p, force_channel=1)
            ct_t, ct_orig = load_img_as_tensor(ct_p, force_channel=1)
        except Exception as e:
            print("Skipping row, cannot read images:", e)
            continue

        enc = TOKENIZER(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        x_t = x_t.to(DEVICE)
        ct_t = ct_t.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        with torch.no_grad():
            logits = model(x_t, ct_t, input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            conf = float(probs[0, pred_idx].item())
            pred_label = "PNEUMONIA" if pred_idx == 1 else "NORMAL"

        # Grad-CAM for xray branch (using model & inputs) -- assumes make_gradcam_heatmap handles xray vs ct by which conv to target
        # Grad-CAM for xray branch
        # -------------------------
        # X-RAY GRADCAM
        # -------------------------
        try:

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
            x_grad_name = f"{uuid.uuid4().hex}.png"

            overlay_heatmap_on_image(x_orig, x_heat, GRADCAM_DIR / x_grad_name)

            heat_resized = cv2.resize(x_heat, (x_orig.shape[1], x_orig.shape[0]))
            mask = heatmap_to_mask(heat_resized, thr=0.4)
            cv2.imwrite(str(SEG_DIR / x_grad_name), mask)

        except Exception as e:
            x_grad_name = ""
            print("xray gradcam failed:", e)

        # -------------------------
        # CT GRADCAM (FIXED: hook layer4, not avgpool)
        # -------------------------
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
            ct_grad_name = f"{uuid.uuid4().hex}.png"

            overlay_heatmap_on_image(ct_orig, ct_heat, CT_GRADCAM_DIR / ct_grad_name)

            ct_resized = cv2.resize(ct_heat, (ct_orig.shape[1], ct_orig.shape[0]))
            ct_mask = heatmap_to_mask(ct_resized, thr=0.4)
            cv2.imwrite(str(CT_SEG_DIR / ct_grad_name), ct_mask)

        except Exception as e:
            ct_grad_name = ""
            print("ct gradcam failed:", e)

        # -------------------------
        # STORE OUTPUT ROW
        # -------------------------
        rows_out.append({
            "xray_path": str(xray_p),
            "ct_path": str(ct_p),
            "clinical_text": text,
            "true_label": label,
            "pred_label": pred_label,
            "confidence": conf,
            "xray_gradcam": str(GRADCAM_DIR / x_grad_name) if x_grad_name else "",
            "xray_seg": str(SEG_DIR / x_grad_name) if x_grad_name else "",
            "ct_gradcam": str(CT_GRADCAM_DIR / ct_grad_name) if ct_grad_name else "",
            "ct_seg": str(CT_SEG_DIR / ct_grad_name) if ct_grad_name else "",
        })


    # write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows_out[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print("Wrote outputs to:", OUT_CSV)

if __name__ == "__main__":
    main()
