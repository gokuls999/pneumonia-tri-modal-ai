#!/usr/bin/env python3
"""
Build labels CSV from the actual downloaded dataset files.

Scans data/raw recursively for image files, assigns:
  - label = 0 for folders named NORMAL (case-insensitive)
  - label = 1 for folders named PNEUMONIA (case-insensitive)
If folder name is different, tries to infer from parent folder names.
Writes CSV to data/raw/labels_auto.csv by default.

Columns: filename,label,text_notes
filename -> basename (just the file name). Ensure your Dataset loader looks in the correct image_dir.
"""

import csv
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}
ROOT = Path("data/raw")
OUT = ROOT / "labels_auto.csv"

def infer_label_from_path(p: Path):
    # check parent folders for NORMAL or PNEUMONIA
    for part in reversed(p.parts):
        name = str(part).lower()
        if "normal" in name:
            return 0
        if "pneumonia" in name or "pneumo" in name:
            return 1
    # fallback: if filename contains 'normal' or 'pneu'
    fname = p.name.lower()
    if "normal" in fname:
        return 0
    if "pneu" in fname or "pneumonia" in fname or "virus" in fname:
        return 1
    return None

def collect_images(root: Path):
    files = []
    for f in root.rglob("*"):
        if f.suffix.lower() in IMG_EXTS and f.is_file():
            files.append(f)
    return files

def main():
    imgs = collect_images(ROOT)
    if not imgs:
        print("No images found under", ROOT)
        return

    rows = []
    for p in imgs:
        label = infer_label_from_path(p)
        if label is None:
            # default to 1 (pneumonia) if unclear — you can change this later
            label = 1
        # use basename so loader can find in given image_dir
        rows.append((p.name, label, "PLACEHOLDER: add clinical notes here"))

    # write CSV (overwrite if exists)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "label", "text_notes"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT}")

if __name__ == "__main__":
    main()
