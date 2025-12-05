import torch
import cv2
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

from src.training.tri_modal_model import TriModalPneumoniaNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = ROOT / "models" / "checkpoints" / "tri_modal_placeholder.pth"

MODEL = TriModalPneumoniaNet()
MODEL.to(DEVICE)
MODEL.eval()

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")

CLASS_MAP = {0: "NORMAL", 1: "PNEUMONIA"}


def preprocess_gray_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img, dtype=torch.float32)


def preprocess_text(text):
    enc = TOKENIZER(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def predict(xray_path, ct_path, clinical_text):
    xray_tensor = preprocess_gray_image(xray_path).to(DEVICE)

    if ct_path is not None:
        ct_tensor = preprocess_gray_image(ct_path).to(DEVICE)
    else:
        ct_tensor = None

    input_ids, attention_mask = preprocess_text(clinical_text)
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    with torch.no_grad():
        logits = MODEL(xray_tensor, ct_tensor, input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred_idx].item()

    return CLASS_MAP[pred_idx], conf
