import sys
from pathlib import Path

from django.http import JsonResponse
from django.views.generic import TemplateView
from django.utils.decorators import method_decorator
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .models import Prediction

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from transformers import AutoTokenizer

# ---------- Project imports ----------
# project root: ...\pneumonia-ai
ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from src.training.multimodal_model import MultimodalPneumoniaNet
from src.explainability.gradcam_image import (
    make_gradcam_heatmap,
    overlay_heatmap_on_image,
)
from src.inference.tri_modal_inference import tri_modal_predict
from src.inference.ct_inference import predict_ct

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load model once
MODEL = MultimodalPneumoniaNet(
    image_backbone_name="densenet121",
    text_model_name="bert-base-uncased",
    text_hidden_size=768,
    fused_hidden_size=512,
    num_classes=2,
)
CKPT_PATH = ROOT / "models" / "checkpoints" / "multimodal_baseline.pth"
MODEL.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
MODEL.to(DEVICE)
MODEL.eval()

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")

CLASS_MAP = {0: "NORMAL", 1: "PNEUMONIA"}


def preprocess_image_file(django_file):
    """
    Read uploaded file -> numpy grayscale image -> model tensor + original image.
    """
    file_bytes = np.frombuffer(django_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")

    orig_img = img.copy()  # keep original for Grad-CAM overlay

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)      # (1, H, W)
    img = np.expand_dims(img, axis=0)      # (B=1, 1, H, W)
    tensor = torch.tensor(img, dtype=torch.float32)
    return tensor, orig_img


def preprocess_text(text, max_len=128):
    enc = TOKENIZER(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


@method_decorator(csrf_exempt, name="dispatch")
class PredictView(APIView):
    authentication_classes = []
    permission_classes = []
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        xray_file = request.FILES.get("xray")
        ct_file = request.FILES.get("ct")
        clinical_text = request.data.get("clinical_text", "")

        if xray_file is None or not clinical_text.strip():
            return JsonResponse(
                {"error": "Both 'xray' file and 'clinical_text' are required."},
                status=400,
            )

        try:
            # ---------- 1) Preprocess X-ray + Text (always needed) ----------
            image_tensor, orig_img = preprocess_image_file(xray_file)
            image_tensor = image_tensor.to(DEVICE)

            input_ids, attention_mask = preprocess_text(clinical_text)
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            # ---------- 2) Base bi-modal prediction (for heatmap + fallback) ----------
            with torch.no_grad():
                logits = MODEL(image_tensor, input_ids, attention_mask)
                probs = F.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                base_conf = probs[0, pred_idx].item()

            # ---------- 3) Grad-CAM heatmap from X-ray + text model ----------
            heatmap = make_gradcam_heatmap(
                MODEL,
                image_tensor,
                input_ids,
                attention_mask,
                device=DEVICE,
                target_class=pred_idx,
            )

            import uuid
            gradcam_name = f"{uuid.uuid4().hex}.png"
            gradcam_dir = Path(settings.MEDIA_ROOT) / "gradcam"
            gradcam_dir.mkdir(parents=True, exist_ok=True)
            gradcam_path = gradcam_dir / gradcam_name

            overlay_heatmap_on_image(orig_img, heatmap, gradcam_path)
            gradcam_url = settings.MEDIA_URL + "gradcam/" + gradcam_name

            # ---------- 4) Decide: use bi-modal OR tri-modal ----------
            final_class = CLASS_MAP[pred_idx]
            final_conf = base_conf
            ct_pred_class = None
            ct_conf = None

            if ct_file is not None:
                # Preprocess CT image (same style as X-ray)
                ct_tensor, _ = preprocess_image_file(ct_file)
                # Run tri-modal fused prediction
                tri_class, tri_conf = tri_modal_predict(
                    image_tensor,  # X-ray tensor
                    ct_tensor,     # CT tensor
                    clinical_text,
                )

                final_class = tri_class
                final_conf = tri_conf
                ct_pred_class = tri_class
                ct_conf = tri_conf

            # ---------- 5) Reset file pointers so Django can save files ----------
            try:
                xray_file.seek(0)
            except Exception:
                pass

            if ct_file is not None:
                try:
                    ct_file.seek(0)
                except Exception:
                    pass

            # ---------- 6) Save prediction history ----------
            prediction = Prediction(
                xray_image=xray_file,
                clinical_text=clinical_text,
                predicted_class=final_class,
                confidence=final_conf,
            )
            # Save Grad-CAM path
            prediction.gradcam_image.name = "gradcam/" + gradcam_name

            # Optional CT fields – adjust names if your model uses different ones
            if ct_file is not None:
                if hasattr(prediction, "ct_image"):
                    prediction.ct_image = ct_file
                if hasattr(prediction, "ct_predicted_class"):
                    prediction.ct_predicted_class = ct_pred_class
                if hasattr(prediction, "ct_confidence"):
                    prediction.ct_confidence = ct_conf

            prediction.save()

            # ---------- 7) JSON response ----------
            return JsonResponse(
                {
                    "predicted_class": final_class,
                    "confidence": final_conf,
                    "clinical_text": clinical_text,
                    "gradcam_url": gradcam_url,
                    "ct_predicted_class": ct_pred_class,
                    "ct_confidence": ct_conf,
                    "used_tri_modal": ct_file is not None,
                }
            )

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


class PredictPageView(TemplateView):
    template_name = "predictor/ui.html"
