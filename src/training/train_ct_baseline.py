import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import models, transforms

from src.dataloaders.ct_dataset import CTCovidDataset

ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = ROOT / "data" / "ct_raw" / "sarscov2"
CKPT_DIR = ROOT / "models" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def build_model():
    """
    ResNet18 adapted for 1-channel CT images.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # change first conv layer to 1-channel
    if model.conv1.in_channels != 1:
        model.conv1 = nn.Conv2d(
            1,
            model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=False,
        )
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # NORMAL vs PNEUMONIA
    return model


def get_dataloaders(batch_size=16, val_ratio=0.2):
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # -> (1,H,W), [0,1]
        ]
    )

    ds = CTCovidDataset(root_dir=DATA_ROOT, transform=tf)
    n_total = len(ds)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


def train_ct(epochs=3, lr=1e-4):
    print(f"Using device: {DEVICE}")
    train_loader, val_loader = get_dataloaders()

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        running_loss = 0.0
        running_corrects = 0
        n_train = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            n_train += images.size(0)

        epoch_loss = running_loss / n_train
        epoch_acc = running_corrects / n_train
        print(f"Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

        # validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        n_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels).item()
                n_val += images.size(0)

        val_epoch_loss = val_loss / n_val
        val_epoch_acc = val_corrects / n_val
        print(f"Val  Loss: {val_epoch_loss:.4f}  Acc: {val_epoch_acc:.4f}")

    ckpt_path = CKPT_DIR / "ct_baseline.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"CT model saved to {ckpt_path}")


if __name__ == "__main__":
    train_ct(epochs=3, lr=1e-4)
