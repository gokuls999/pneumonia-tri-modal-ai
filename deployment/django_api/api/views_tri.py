# deployment/django_api/api/views_tri.py

import json
import uuid
from pathlib import Path

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings

import torch
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# make sure project root is importable
import sys
REPO_ROOT = Path(__file__).resolve().parents[3]   # .../pneumonia-ai
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.inference.tri_infer_core import tri_infer_single


###############################################################################
# ORIGINAL ENDPOINT — KEEPING EXACTLY AS IS
###############################################################################

@csrf_exempt
def tri_infer_view(request):
    """
    POST JSON:
    {
        "xray_path": "data/raw/chest_xray/....jpeg",
        "ct_path": "data/ct_raw/sarscov2/....png",
        "text": "clinical text here"
    }
    """
    if request.method != "POST":
        return JsonResponse({"detail": "Use POST"}, status=405)

    # ---- parse JSON body ----
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"detail": "Invalid JSON body"}, status=400)

    xray_path = body.get("xray_path")
    ct_path = body.get("ct_path")
    text = body.get("text") or body.get("clinical_text")

    if not xray_path or not ct_path or not text:
        return JsonResponse(
            {
                "detail": "Required fields: xray_path, ct_path, text (or clinical_text)"
            },
            status=400,
        )

    # ---- resolve relative paths from repo root ----
    def resolve_path(p: str) -> Path:
        p = Path(p)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p

    xray_abs = resolve_path(xray_path)
    ct_abs = resolve_path(ct_path)

    if not xray_abs.exists():
        return JsonResponse({"detail": f"xray file not found: {xray_abs}"}, status=400)
    if not ct_abs.exists():
        return JsonResponse({"detail": f"ct file not found: {ct_abs}"}, status=400)

    # ---- run tri-modal inference core ----
    try:
        result = tri_infer_single(
            xray_path=str(xray_abs),
            ct_path=str(ct_abs),
            clinical_text=text,
        )
    except Exception as e:
        return JsonResponse(
            {"detail": "Inference failed", "error": str(e)}, status=500
        )

    # ---- convert any local filesystem media paths -> media URLs for browser ----
    # helper: filesystem path under settings.MEDIA_ROOT -> settings.MEDIA_URL + relpath
    from django.conf import settings

    def fs_to_media_url(path_str: str) -> str:
        if not path_str:
            return ""
        s = str(path_str).replace("\\", "/")  # normalize windows separators
        media_root = str(settings.MEDIA_ROOT).replace("\\", "/").rstrip("/")
        media_url_base = str(settings.MEDIA_URL).rstrip("/")

        # If the returned path lies under MEDIA_ROOT, convert to MEDIA_URL
        if s.startswith(media_root):
            rel = s[len(media_root):].lstrip("/")
            return media_url_base + "/" + rel

        # If already a MEDIA URL (starts with /media or equivalent), return it unchanged
        if s.startswith(settings.MEDIA_URL) or s.startswith(media_url_base + "/"):
            return s

        # Fallback: if it's an absolute FS path inside the project, try to detect "/media/" substring
        parts = s.split("/")
        if "media" in parts:
            i = parts.index("media")
            rel = "/".join(parts[i + 1 :])
            return media_url_base + "/" + rel

        # Otherwise return normalized string (may be absolute FS path) — let browser attempt to load
        return s

    # replace the four keys if present
    for k in ("xray_gradcam", "xray_segmentation", "ct_gradcam", "ct_segmentation"):
        if k in result:
            result[k] = fs_to_media_url(result[k])

    # --- DEBUG: include filesystem checks so we can see why /media/... returns 404 ---
    debug = {}
    media_root_p = Path(settings.MEDIA_ROOT)
    for k in ("xray_gradcam", "xray_segmentation", "ct_gradcam", "ct_segmentation"):
        returned = result.get(k) or ""
        # canonical URL we returned
        debug[f"{k}_url"] = returned

        # compute the filesystem path we expect for this URL
        if returned.startswith(str(settings.MEDIA_URL)) or returned.startswith(str(settings.MEDIA_URL).rstrip("/")):
            rel = returned[len(str(settings.MEDIA_URL)) :].lstrip("/").replace("/", Path.sep)
            fs_path = media_root_p / rel
        elif returned.startswith(str(settings.MEDIA_URL).rstrip("/")):
            rel = returned[len(str(settings.MEDIA_URL).rstrip("/")) :].lstrip("/").replace("/", Path.sep)
            fs_path = media_root_p / rel
        else:
            # treat returned value as FS path (already normalized above)
            fs_path = Path(returned)

        debug[f"{k}_fs"] = str(fs_path)
        debug[f"{k}_exists"] = fs_path.exists()

    result["_debug_media"] = debug




###############################################################################
# NEW ENDPOINT 1:
# TRI-MODAL INFERENCE WITH FILE UPLOAD
###############################################################################

@csrf_exempt
@require_http_methods(["POST"])
def tri_infer_upload_view(request):
    """
    Accepts multipart upload:
        xray_file: uploaded X-ray image
        ct_file:   uploaded CT image
        text:      clinical text

    Saves files into MEDIA_ROOT/uploads/<uuid>/...
    Returns inference result identical to tri_infer_view.
    """

    xray_file = request.FILES.get("xray_file")
    ct_file = request.FILES.get("ct_file")
    text = request.POST.get("text") or request.POST.get("clinical_text") or ""

    if not xray_file or not ct_file:
        return JsonResponse({"detail": "Missing xray_file or ct_file"}, status=400)

    # Where uploaded files go
    upload_dir = Path(settings.MEDIA_ROOT) / "uploads" / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)

    xray_path = upload_dir / xray_file.name
    ct_path = upload_dir / ct_file.name

    # Save uploaded files
    with open(xray_path, "wb") as f:
        for chunk in xray_file.chunks():
            f.write(chunk)

    with open(ct_path, "wb") as f:
        for chunk in ct_file.chunks():
            f.write(chunk)

    # Run inference
    try:
        result = tri_infer_single(
            xray_path=str(xray_path),
            ct_path=str(ct_path),
            clinical_text=text,
        )
    except Exception as e:
        return JsonResponse({"detail": "Inference failed", "error": str(e)}, status=500)

    return JsonResponse(result, safe=False)



###############################################################################
# DEMO SCORE GENERATOR — OPTION B
###############################################################################

def simulate_scores():
    import random
    random.seed(uuid.uuid4().int & 0xFFFF)

    bacterial = round(random.uniform(0.05, 0.6), 2)
    viral = round(random.uniform(0.05, 0.9 - bacterial), 2)
    severity = round(random.uniform(0.1, 0.8), 2)
    ventilation = round(min(1.0, severity * random.uniform(0.4, 1.2)), 2)
    mortality = round(min(1.0, severity * random.uniform(0.02, 0.2)), 3)

    return {
        "bacterial_prob": bacterial,
        "viral_prob": viral,
        "severity_score": severity,
        "ventilation_risk": ventilation,
        "mortality_risk": mortality,
    }



###############################################################################
# NEW ENDPOINT 2:
# GENERATE PDF REPORT (SIMULATED ETIOLOGY + SEVERITY)
###############################################################################

# Replaces existing generate_report_view in deployment/django_api/api/views_tri.py
@csrf_exempt
@require_http_methods(["POST"])
def generate_report_view(request):
    """
    POST payload:
      { "inference": { ... }, "simulate": true|false }

    Writes PDF to MEDIA_ROOT/reports/<uuid>.pdf and returns:
      { "pdf_url": "/media/reports/<uuid>.pdf" }
    """
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    inference = body.get("inference")
    simulate = bool(body.get("simulate", True))

    if not isinstance(inference, dict):
        return JsonResponse({"error": "Missing inference object"}, status=400)

    # If the caller didn't provide explicit scores, simulate demo scores
    scores = inference.get("scores") if isinstance(inference.get("scores"), dict) else simulate_scores() if simulate else {}

    reports_dir = Path(settings.MEDIA_ROOT) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_name = f"report_{uuid.uuid4().hex}.pdf"
    pdf_path = reports_dir / pdf_name

    # helper: map inference image value -> filesystem path (if under MEDIA_URL) or leave if absolute fs path
    def resolve_image_path(img_val):
        if not img_val:
            return None
        s = str(img_val)
        # Already absolute fs path?
        p = Path(s)
        if p.exists():
            return p
        # If returned value looks like a media URL (/media/...), map to MEDIA_ROOT
        media_url = str(settings.MEDIA_URL).rstrip("/")
        try:
            if s.startswith(media_url):
                rel = s[len(media_url):].lstrip("/")
                candidate = Path(settings.MEDIA_ROOT) / rel
                if candidate.exists():
                    return candidate
            # some code returns '/media/...' with leading slash, handle that
            if s.startswith("/" + media_url.lstrip("/")):
                rel = s.split("/", 2)[-1] if "/" in s[1:] else None
                if rel:
                    candidate = Path(settings.MEDIA_ROOT) / rel
                    if candidate.exists():
                        return candidate
            # or image string is windows path with backslashes produced earlier -> normalize
            s2 = s.replace("\\", "/")
            if "/media/" in s2:
                rel = s2.split("/media/", 1)[1]
                candidate = Path(settings.MEDIA_ROOT) / rel
                if candidate.exists():
                    return candidate
        except Exception:
            pass
        return None

    # PDF layout helpers
    try:
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        W, H = A4
        left = 40
        right_margin = 40
        usable_w = W - left - right_margin
        top = H - 40
        y = top

        # Header: hospital-like
        c.setFont("Helvetica-Bold", 18)
        c.drawString(left, y, "AI Pneumonia Diagnostic Assistant")
        c.setFont("Helvetica", 10)
        # small line with patient id/time/model if present
        hdr_parts = []
        pid = inference.get("patient_id") or inference.get("patientID") or "PATIENT_ID"
        study_time = inference.get("analysis_time") or inference.get("study_time") or inference.get("timestamp")
        if study_time:
            hdr_parts.append(str(study_time))
        hdr_parts.append(f"Patient: {pid}")
        model_name = inference.get("model_name") or inference.get("model") or "Model: PneumoAI"
        hdr_right = " | ".join([p for p in [model_name] if p])
        # right aligned small text
        c.setFont("Helvetica", 9)
        c.drawRightString(W - right_margin, y, hdr_right)
        y -= 22

        # divider line
        c.setLineWidth(0.8)
        c.line(left, y, W - right_margin, y)
        y -= 18

        # FINAL DIAGNOSIS block
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "FINAL AI DIAGNOSIS")
        y -= 16
        c.setFont("Helvetica", 11)
        pred = inference.get("predicted_label", "N/A")
        conf = inference.get("confidence", None)
        conf_text = f" (confidence {conf * 100:.1f}%)" if isinstance(conf, (float, int)) else ""
        c.drawString(left + 6, y, f"{pred}{conf_text}")
        y -= 18

        # Etiology table: bacterial/viral/normal probabilities if present or simulated
        # Build rows: prefer inference['etiology'] if provided (dict), else use 'scores' or simulated
        etio = inference.get("etiology") if isinstance(inference.get("etiology"), dict) else None
        if not etio:
            # infer from available keys
            etio = {
                "Bacterial": inference.get("bacterial_prob") or scores.get("bacterial_prob") or 0.0,
                "Viral": inference.get("viral_prob") or scores.get("viral_prob") or 0.0,
                "Normal/Other": inference.get("normal_prob") or round(1.0 - (inference.get("bacterial_prob",0) + inference.get("viral_prob",0)), 2) if inference.get("bacterial_prob") is not None else scores.get("normal_prob", 0.0),
            }

        # table header
        c.setFont("Helvetica-Bold", 11)
        c.drawString(left + 6, y, "Etiology")
        c.drawRightString(W - right_margin - 6, y, "Probability")
        y -= 14
        c.setLineWidth(0.4)
        c.line(left + 6, y, W - right_margin - 6, y)
        y -= 10
        c.setFont("Helvetica", 10)
        for k, v in etio.items():
            prob = float(v) if v is not None else 0.0
            c.drawString(left + 8, y, str(k))
            c.drawRightString(W - right_margin - 6, y, f"{prob * 100:.1f}%" if prob <= 1.0 else f"{prob:.1f}%")
            y -= 14
        y -= 10

        # Progression & severity table (use scores dict)
        prog_keys = [
            ("severity_score", "Current Severity"),
            ("ventilation_risk", "Probability of requiring mechanical ventilation"),
            ("mortality_risk", "Predicted mortality risk (7-day)"),
        ]
        if scores:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(left + 6, y, "Progression & Severity (Demo)")
            y -= 14
            c.setFont("Helvetica", 10)
            for key, label in prog_keys:
                val = scores.get(key, None)
                if val is None:
                    continue
                # render like "71%  (High)"
                if isinstance(val, float):
                    perc = val * 100 if val <= 1.0 else val
                    perc_text = f"{perc:.0f}%"
                else:
                    perc_text = str(val)
                # risk level heuristic
                risk_level = "Low"
                if isinstance(val, (int, float)):
                    v100 = val * 100 if val <= 1.0 else val
                    if v100 >= 80:
                        risk_level = "Very High"
                    elif v100 >= 60:
                        risk_level = "High"
                    elif v100 >= 30:
                        risk_level = "Moderate"
                    else:
                        risk_level = "Low"
                c.drawString(left + 8, y, label)
                c.drawRightString(W - right_margin - 6, y, f"{perc_text}  ({risk_level})")
                y -= 14
            y -= 6

        # clinical text block (wrap)
        clinical = inference.get("clinical_text") or inference.get("text") or ""
        if clinical:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(left + 6, y, "Clinical text")
            y -= 14
            c.setFont("Helvetica", 9)
            from reportlab.lib.utils import simpleSplit
            wrapped = simpleSplit(str(clinical), "Helvetica", 9, usable_w)
            for ln in wrapped[:6]:
                c.drawString(left + 8, y, ln)
                y -= 12
            y -= 10

        # Draw a separator before images
        c.setLineWidth(0.6)
        c.line(left, y, W - right_margin, y)
        y -= 14

        # Images area: arrange two images per row (xray gradcam + segmentation), (ct gradcam + segmentation)
        image_keys = [
            ("xray_gradcam", "X-ray Grad-CAM"),
            ("xray_segmentation", "X-ray Segmentation"),
            ("ct_gradcam", "CT Grad-CAM"),
            ("ct_segmentation", "CT Segmentation"),
        ]
        img_max_w = (usable_w - 12) / 2.0  # two columns with small gap
        img_max_h = 160

        # helper to draw one image card: caption above, scaled image below
        def draw_img_card(img_path_obj: Path, caption: str, x_pos: float, y_pos: float):
            nonlocal c
            if not img_path_obj or not img_path_obj.exists():
                return y_pos
            try:
                c.setFont("Helvetica", 9)
                c.drawString(x_pos, y_pos, caption)
                y_img_top = y_pos - 12
                # compute sizing preserving aspect ratio
                reader = ImageReader(str(img_path_obj))
                iw, ih = reader.getSize()
                scale = min(img_max_w / iw, img_max_h / ih, 1.0)
                draw_w = iw * scale
                draw_h = ih * scale
                c.drawImage(reader, x_pos, y_img_top - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True)
                return y_img_top - draw_h - 8
            except Exception:
                return y_pos

        # iterate pairs and place
        x_left = left + 6
        x_right = left + 6 + img_max_w + 12
        cur_y = y
        col = 0
        max_row_bottom = cur_y
        for key, caption in image_keys:
            img_p = resolve_image_path(inference.get(key))
            if col == 0:
                # left column
                new_y = draw_img_card(img_p, caption, x_left, cur_y)
                if new_y < max_row_bottom:
                    max_row_bottom = new_y
                col = 1
            else:
                new_y_r = draw_img_card(img_p, caption, x_right, cur_y)
                if new_y_r < max_row_bottom:
                    max_row_bottom = new_y_r
                # row finished -> move cur_y down to max_row_bottom - gap
                cur_y = max_row_bottom - 18
                # reset for next row
                max_row_bottom = cur_y
                col = 0
            # if not enough space on page, make a new page
            if cur_y < 120:
                c.showPage()
                cur_y = top
                max_row_bottom = cur_y
                col = 0

        # finalize
        c.showPage()
        c.save()
    except Exception as e:
        return JsonResponse({"error": f"PDF generation failed: {e}"}, status=500)

    pdf_url = f"{settings.MEDIA_URL.rstrip('/')}/reports/{pdf_name}"
    return JsonResponse({"pdf_url": pdf_url})

