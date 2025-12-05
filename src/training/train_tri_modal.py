import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tqdm import tqdm

# --- project root ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.dataloaders.tri_modal_dataset import TriModalDataset
from src.training.tri_modal_model import TriModalPneumoniaNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 2   # you can increase later
LR = 1e-4

CSV_PATH = ROOT / "data" / "processed" / "tri_modal_pairs.csv"
DATA_ROOT = ROOT / "data"

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")


def collate_fn(batch):
    xray_imgs, ct_imgs, texts, labels = zip(*batch)

    xray_imgs = torch.stack(xray_imgs, dim=0)   # (B, 1, 224, 224)
    ct_imgs = torch.stack(ct_imgs, dim=0)       # (B, 1, 224, 224)
    labels = torch.stack(labels, dim=0)         # (B,)

    enc = TOKENIZER(
        list(texts),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    return xray_imgs, ct_imgs, input_ids, attention_mask, labels


def main():
    print("Using device:", DEVICE)

    dataset = TriModalDataset(
        csv_file=CSV_PATH,
        xray_root=DATA_ROOT,
        ct_root=DATA_ROOT,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = TriModalPneumoniaNet(
        text_model_name="bert-base-uncased",
        num_classes=2,
        fused_hidden_size=512,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for xray_imgs, ct_imgs, input_ids, attention_mask, labels in pbar:
            xray_imgs = xray_imgs.to(DEVICE)
            ct_imgs = ct_imgs.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(xray_imgs, ct_imgs, input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{running_loss / total:.4f}",
                "acc": f"{correct / total:.4f}",
            })

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f} Acc={epoch_acc:.4f}")

    ckpt_path = ROOT / "models" / "checkpoints" / "tri_modal_baseline.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print("Tri-modal model saved to:", ckpt_path)


if __name__ == "__main__":
    main()
