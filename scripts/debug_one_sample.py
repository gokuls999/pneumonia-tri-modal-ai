# scripts/debug_one_sample.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # <-- project root
sys.path.append(str(ROOT))

from pathlib import Path
import numpy as np, cv2, torch
from transformers import AutoTokenizer
from src.inference.eval_tri_inference import load_img_as_tensor, TOKENIZER, ROOT
from src.training.tri_modal_model import TriModalPneumoniaNet
from src.explainability.gradcam_image import make_gradcam_heatmap

ROOT = Path.cwd()
CSV = ROOT / "data" / "processed" / "tri_dataset_v2_val.csv"

# pick first non-empty row from CSV
import csv
with open(CSV, newline="", encoding="utf-8") as f:
    r = next(csv.DictReader(f))

xray_p = Path(r["xray_path"])
ct_p = Path(r["ct_path"])
text = r["clinical_text"]

print("Sample:", xray_p, ct_p)
x_t, x_orig = load_img_as_tensor(xray_p, force_channel=1)
ct_t, ct_orig = load_img_as_tensor(ct_p, force_channel=1)
enc = AutoTokenizer.from_pretrained("bert-base-uncased")(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TriModalPneumoniaNet().to(device)
model.load_state_dict(torch.load(ROOT / "models" / "checkpoints" / "tri_modal_baseline.pth", map_location=device))
model.eval()

x_t = x_t.to(device); ct_t = ct_t.to(device); input_ids = input_ids.to(device); attention_mask = attention_mask.to(device)

print("Calling make_gradcam_heatmap for CT...")
ct_heat = make_gradcam_heatmap(model, None, ct_t, input_ids, attention_mask, device=device, backbone=model.ct_backbone, branch="ct")
print("ct_heat type:", type(ct_heat), "shape (np):", getattr(ct_heat, "shape", None), "min/max:", (np.nanmin(ct_heat) if isinstance(ct_heat, np.ndarray) else None, np.nanmax(ct_heat) if isinstance(ct_heat, np.ndarray) else None))
print("Done.")
