import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer
from src.training.tri_modal_model import TriModalPneumoniaNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = ROOT / "models" / "checkpoints" / "tri_modal_baseline.pth"

CLASS_MAP = {0: "NORMAL", 1: "PNEUMONIA"}

# Load model
MODEL = TriModalPneumoniaNet()
MODEL.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE), strict=False)
MODEL.to(DEVICE)
MODEL.eval()

# Tokenizer
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")


def tri_modal_predict(xray_tensor, ct_tensor, clinical_text):
    xray_tensor = xray_tensor.to(DEVICE)
    ct_tensor = ct_tensor.to(DEVICE)

    enc = TOKENIZER(
        clinical_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = MODEL(xray_tensor, ct_tensor, input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred_idx].item()

    return CLASS_MAP[pred_idx], conf
