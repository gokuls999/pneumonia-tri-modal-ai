import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import torch
import cv2
import numpy as np
from transformers import AutoTokenizer

from src.training.multimodal_model import MultimodalPneumoniaNet


def preprocess_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)      # (1, H, W)
    img = np.expand_dims(img, axis=0)      # (B=1, 1, H, W)
    return torch.tensor(img, dtype=torch.float32)


def preprocess_text(text, tokenizer, max_len=128):
    encoded = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return encoded["input_ids"], encoded["attention_mask"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 🔁 CHANGE THIS to any valid image from your dataset:
    image_path = "data/raw/chest_xray/chest_xray/test/PNEUMONIA/person11_virus_38.jpeg"

    # 🔁 You can change this clinical text anytime:
    clinical_text = "High fever, cough, shortness of breath for 3 days."

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load model
    model = MultimodalPneumoniaNet(
        image_backbone_name="densenet121",
        text_model_name="bert-base-uncased",
        text_hidden_size=768,
        fused_hidden_size=512,
        num_classes=2,
    )

    ckpt_path = "models/checkpoints/multimodal_baseline.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess inputs
    image_tensor = preprocess_image(image_path).to(device)
    input_ids, attention_mask = preprocess_text(clinical_text, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Inference
    with torch.no_grad():
        logits = model(image_tensor, input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    class_map = {0: "NORMAL", 1: "PNEUMONIA"}

    print("\n--- Prediction Result ---")
    print("Image:", image_path)
    print("Clinical text:", clinical_text)
    print("Predicted class:", class_map[pred])
    print("Confidence:", probs[0][pred].item())


if __name__ == "__main__":
    main()
