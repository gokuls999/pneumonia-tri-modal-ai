import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.training.multimodal_model import MultimodalPneumoniaNet
from src.inference.multimodal_infer import preprocess_image


def explain_text_importance(
    image_path: str,
    clinical_text: str,
    model_ckpt: str = "models/checkpoints/multimodal_baseline.pth",
    text_model_name: str = "bert-base-uncased",
    max_len: int = 128,
    top_k: int = 10,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # load model
    model = MultimodalPneumoniaNet(
        image_backbone_name="densenet121",
        text_model_name=text_model_name,
        text_hidden_size=768,
        fused_hidden_size=512,
        num_classes=2,
    )
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.to(device)
    model.eval()

    # preprocess image
    image_tensor = preprocess_image(image_path).to(device)

    # tokenize text
    enc = tokenizer(
        clinical_text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)          # (1,L)
    attention_mask = enc["attention_mask"].to(device)

    # original prediction
    with torch.no_grad():
        logits = model(image_tensor, input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        orig_prob = probs[0, pred_class].item()

    class_map = {0: "NORMAL", 1: "PNEUMONIA"}
    print(f"\nPredicted class: {class_map[pred_class]}  (p = {orig_prob:.4f})")

    # we will mask tokens one by one and see probability drop
    input_ids_cpu = input_ids[0].detach().cpu().clone()
    tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu)

    mask_id = tokenizer.mask_token_id

    importance = []

    for i, tok_id in enumerate(input_ids_cpu):
        tok = tokens[i]

        # skip special tokens and padding
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # create a copy and mask position i
        modified_ids = input_ids_cpu.clone()
        modified_ids[i] = mask_id
        modified_ids_batch = modified_ids.unsqueeze(0).to(device)

        with torch.no_grad():
            logits_mod = model(image_tensor, modified_ids_batch, attention_mask)
            probs_mod = F.softmax(logits_mod, dim=1)
            prob_target = probs_mod[0, pred_class].item()

        # importance = drop in probability when this token is masked
        delta = orig_prob - prob_target
        importance.append((i, tokens[i], delta))

    # sort by importance
    importance.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop {top_k} important tokens:")
    for i, tok, score in importance[:top_k]:
        # clean up '##' subwords visually
        clean_tok = tok.replace("##", "")
        print(f"{clean_tok:15s}  Δp = {score:.4f}")


def main():
    # same defaults as before – change image and text if you like
    image_path = "data/raw/chest_xray/chest_xray/test/PNEUMONIA/person11_virus_38.jpeg"
    clinical_text = "High fever, cough, shortness of breath for 3 days."

    explain_text_importance(
        image_path=image_path,
        clinical_text=clinical_text,
        model_ckpt="models/checkpoints/multimodal_baseline.pth",
        text_model_name="bert-base-uncased",
        max_len=128,
        top_k=10,
    )


if __name__ == "__main__":
    main()
