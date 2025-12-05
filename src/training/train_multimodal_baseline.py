import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataloaders.xray_text_dataset import XrayTextDataset
from src.training.multimodal_model import MultimodalPneumoniaNet


def get_dataloaders(batch_size=8):
    train_ds = XrayTextDataset(
        image_root="data/raw/chest_xray",
        labels_csv="data/raw/labels_auto.csv",
        tokenizer_name="bert-base-uncased",
        max_len=128,
    )
    # for now, just use a subset for quick testing
    indices = list(range(len(train_ds)))
    subset_indices = indices[:512]  # small subset
    subset = torch.utils.data.Subset(train_ds, subset_indices)

    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Training", ncols=80):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images, input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    loader = get_dataloaders(batch_size=4)

    model = MultimodalPneumoniaNet(
        image_backbone_name="densenet121",
        text_model_name="bert-base-uncased",
        text_hidden_size=768,
        fused_hidden_size=512,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    num_epochs = 1  # start with 1 for quick test

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)
        print(f"Loss: {loss:.4f}  Acc: {acc:.4f}")

    # save checkpoint
    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "models/checkpoints/multimodal_baseline.pth")
    print("Model saved to models/checkpoints/multimodal_baseline.pth")


if __name__ == "__main__":
    main()
