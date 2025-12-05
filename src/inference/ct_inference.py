import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------- Paths & device ----------
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = ROOT / "models" / "checkpoints" / "ct_baseline.pth"

CLASS_MAP = {0: "NORMAL", 1: "PNEUMONIA"}


def build_ct_model():
    """
    Must match the architecture used in train_ct_baseline.py
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


# load once at import
CT_MODEL = build_ct_model()
CT_MODEL.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
CT_MODEL.to(DEVICE)
CT_MODEL.eval()

# preprocessing pipeline for a single CT image
CT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # -> (C,H,W)
        # if image is RGB (3-channel), convert to 1-channel by averaging
        transforms.Lambda(lambda t: t.mean(dim=0, keepdim=True)),
    ]
)


def preprocess_ct_image(image_path: Path | str) -> torch.Tensor:
    """
    Load CT image from disk and convert to (1, 1, 224, 224) tensor.
    """
    img = Image.open(image_path).convert("L")  # grayscale
    img = CT_TRANSFORM(img)  # (1, 224, 224)
    img = img.unsqueeze(0)   # (B=1, 1, 224, 224)
    return img.to(DEVICE)


def predict_ct(image_path: Path | str):
    """
    Run CT model on a single image path.

    Returns:
        predicted_class (str), confidence (float)
    """
    x = preprocess_ct_image(image_path)

    with torch.no_grad():
        logits = CT_MODEL(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred_idx].item()

    return CLASS_MAP[pred_idx], conf


if __name__ == "__main__":
    # quick manual test (optional):
    # python src/inference/ct_inference.py
    sample = ROOT / "data" / "ct_raw" / "sarscov2" / "PNEUMONIA"
    if sample.exists():
        any_img = next(sample.rglob("*.png"), None) or next(sample.rglob("*.jpg"), None)
        if any_img:
            label, c = predict_ct(any_img)
            print("Sample:", any_img)
            print("Predicted:", label, "confidence:", c)
        else:
            print("No CT images found in PNEUMONIA folder for test.")
    else:
        print("Sample folder not found:", sample)
