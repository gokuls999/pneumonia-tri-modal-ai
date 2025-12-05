import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.dataloaders.xray_text_dataset import XrayTextDataset

if __name__ == "__main__":
    ds = XrayTextDataset(
        image_root="data/raw/chest_xray",
        labels_csv="data/raw/labels_auto.csv",
        tokenizer_name="bert-base-uncased",
        max_len=128,
    )
    print("Dataset size:", len(ds))
    sample = ds[0]
    print("image:", sample["image"].shape)
    print("input_ids:", sample["input_ids"].shape)
    print("label:", sample["label"])
