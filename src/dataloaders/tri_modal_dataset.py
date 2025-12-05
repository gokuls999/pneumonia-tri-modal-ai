import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class TriModalDataset(Dataset):
    """
    Returns:
      xray_img: tensor (1, 224, 224)
      ct_img:   tensor (1, 224, 224)
      text:     string
      label:    tensor(int64)
    """
    def __init__(self, csv_file, xray_root, ct_root):
        self.df = pd.read_csv(csv_file)

        self.xray_root = Path(xray_root)
        self.ct_root = Path(ct_root)

        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        xray_path = self.xray_root / row["xray_path"]
        ct_path = self.ct_root / row["ct_path"]
        text = row["clinical_text"]
        label = int(row["label"])

        xray_img = self.transform(Image.open(xray_path))
        ct_img = self.transform(Image.open(ct_path))

        return xray_img, ct_img, text, torch.tensor(label)
