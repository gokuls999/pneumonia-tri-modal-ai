import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.training.multimodal_model import MultimodalPneumoniaNet


def preprocess_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    orig = img.copy()

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)      # (1,H,W)
    img = np.expand_dims(img, axis=0)      # (B=1,1,H,W)
    tensor = torch.tensor(img, dtype=torch.float32)
    return tensor, orig


def preprocess_text(text, tokenizer, max_len=128):
    enc = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def make_gradcam_heatmap(model, image_tensor, input_ids, attention_mask, device, target_class=None):
    """
    Grad-CAM over the last conv feature map of the image backbone (DenseNet).
    """

    feats = None
    grads = None

    def fwd_hook(module, inp, out):
        nonlocal feats
        feats = out

    def bwd_hook(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0]

    handle_fwd = model.image_backbone.features.register_forward_hook(fwd_hook)
    handle_bwd = model.image_backbone.features.register_backward_hook(bwd_hook)

    image_tensor = image_tensor.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    model.zero_grad()
    logits = model(image_tensor, input_ids, attention_mask)  # (1,C)

    if target_class is None:
        target_class = torch.argmax(logits, dim=1).item()

    loss = logits[0, target_class]
    loss.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    # feats: (1, C, H, W)
    # grads: (1, C, H, W)
    weights = grads.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
    cam = (weights * feats).sum(dim=1, keepdim=True)  # (1,1,H,W)
    cam = F.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()  # (H,W)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    return cam  # 0–1


def overlay_heatmap_on_image(orig_img, heatmap, out_path):
    h, w = orig_img.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))

    heatmap_color = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(orig_rgb, 0.6, heatmap_color, 0.4, 0)

    cv2.imwrite(str(out_path), overlay)
    return out_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 🔁 same image & text as inference (change if you like)
    image_path = Path("data/raw/chest_xray/chest_xray/test/PNEUMONIA/person11_virus_38.jpeg")
    clinical_text = "High fever, cough, shortness of breath for 3 days."

    # preprocess
    image_tensor, orig = preprocess_image(image_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    input_ids, attention_mask = preprocess_text(clinical_text, tokenizer)

    # load model
    model = MultimodalPneumoniaNet(
        image_backbone_name="densenet121",
        text_model_name="bert-base-uncased",
        text_hidden_size=768,
        fused_hidden_size=512,
        num_classes=2,
    )
    ckpt = "models/checkpoints/multimodal_baseline.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    # grad-cam
    heatmap = make_gradcam_heatmap(
        model,
        image_tensor,
        input_ids,
        attention_mask,
        device=device,
        target_class=None,  # uses predicted class
    )

    out_dir = Path("models/exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gradcam_person11_virus_38.png"
    overlay_heatmap_on_image(orig, heatmap, out_path)

    print("Grad-CAM saved to:", out_path)


if __name__ == "__main__":
    main()
