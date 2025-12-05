import random
import pandas as pd
from pathlib import Path

# -------- CONFIG --------
XRAY_ROOT = Path("data/raw/chest_xray/chest_xray")
CT_ROOT = Path("data/ct_raw/sarscov2")
OUTPUT_CSV = Path("data/processed/tri_modal_pairs.csv")

TEXT_TEMPLATES = {
    0: [
        "No fever, normal breathing, healthy condition.",
        "No signs of infection or lung disease.",
    ],
    1: [
        "High fever, cough, shortness of breath.",
        "Severe respiratory distress and chest pain.",
    ],
}

# -------- HELPERS --------
def collect_images(root):
    samples = []
    for cls in ["NORMAL", "PNEUMONIA"]:
        for p in (root / "train" / cls).glob("*.jpeg"):
            samples.append((str(p), 0 if cls == "NORMAL" else 1))
        for p in (root / "train" / cls).glob("*.png"):
            samples.append((str(p), 0 if cls == "NORMAL" else 1))
    return samples


def collect_ct_images(root):
    samples = []
    for cls in ["NORMAL", "PNEUMONIA"]:
        for p in (root / cls).glob("*.png"):
            samples.append((str(p), 0 if cls == "NORMAL" else 1))
    return samples


# -------- BUILD PAIRS --------
xray_samples = collect_images(XRAY_ROOT)
ct_samples = collect_ct_images(CT_ROOT)

rows = []

for _ in range(min(len(xray_samples), len(ct_samples))):
    xray_path, label = random.choice(xray_samples)
    ct_path, _ = random.choice(ct_samples)

    text = random.choice(TEXT_TEMPLATES[label])

    rows.append({
        "xray_path": Path(xray_path).relative_to("data").as_posix(),
        "ct_path": Path(ct_path).relative_to("data").as_posix(),
        "clinical_text": text,
        "label": label,
    })

df = pd.DataFrame(rows)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved {len(df)} tri-modal samples to:", OUTPUT_CSV)
