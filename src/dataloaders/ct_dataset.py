from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image


class CTCovidDataset(Dataset):
    """
    Simple CT dataset for binary classification using the renamed folders:

    data/ct_raw/sarscov2/NORMAL
    data/ct_raw/sarscov2/PNEUMONIA

    label: 0 = NORMAL, 1 = PNEUMONIA
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []
        classes = {"NORMAL": 0, "PNEUMONIA": 1}

        for cls_name, label in classes.items():
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.rglob("*.png"):
                self.samples.append((img_path, label))
            for img_path in cls_dir.rglob("*.jpg"):
                self.samples.append((img_path, label))
            for img_path in cls_dir.rglob("*.jpeg"):
                self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(f"No CT images found under {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")  # grayscale

        if self.transform is not None:
            img = self.transform(img)
        else:
            # default: resize to 224x224 and convert to tensor in [0,1]
            from torchvision import transforms

            default_tf = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),  # (C,H,W) with C=1
                ]
            )
            img = default_tf(img)

        return img, torch.tensor(label, dtype=torch.long)
