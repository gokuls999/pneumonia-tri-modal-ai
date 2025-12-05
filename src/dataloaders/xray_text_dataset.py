import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from transformers import AutoTokenizer


class XrayTextDataset(Dataset):
    """
    Multimodal dataset: Chest X-ray + Clinical Text
    Uses labels_auto.csv (filename,label,text_notes)
    and searches for images under image_root recursively.
    """

    def __init__(self, image_root, labels_csv,
                 tokenizer_name="bert-base-uncased",
                 max_len=128):
        import pandas as pd

        self.image_root = Path(image_root)
        self.data = pd.read_csv(labels_csv)

        # build map: filename -> full path (search all subfolders)
        self.path_map = {}
        for p in self.image_root.rglob("*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                self.path_map[p.name] = p

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def load_xray(self, filename):
        if filename not in self.path_map:
            raise FileNotFoundError(f"Image file not found for: {filename}")

        path = self.path_map[filename]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")

        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1,H,W)
        return torch.tensor(img, dtype=torch.float32)

    def encode_text(self, text):
        encoded = self.tokenizer(
            str(text),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        filename = row["filename"]
        label = int(row["label"])
        text = row["text_notes"]

        image_tensor = self.load_xray(filename)
        input_ids, attention_mask = self.encode_text(text)

        return {
            "image": image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }
