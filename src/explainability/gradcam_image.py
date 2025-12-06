# src/explainability/gradcam_image.py
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def make_gradcam_heatmap(
    model,
    xray_tensor,            # Tensor or None - (1, C, H, W)
    ct_tensor,              # Tensor or None - (1, C, H, W)
    input_ids,              # Tensor (1, L) or None (pass torch.zeros(1,1,dtype=torch.long) if None)
    attention_mask,         # Tensor (1, L) or None
    device="cpu",
    target_class=None,
    backbone=None,          # the module to hook (nn.Module)
    branch="xray",          # "xray" or "ct" - which backbone we hooked
):
    """
    Debuggable and robust Grad-CAM helper.

    Notes:
    - We clone/detach captured features and grads to avoid 'view+inplace' autograd issues.
    - This function prints debug info (shapes, dtypes, min/max) to the console.
    """

    import torch.nn.functional as F

    if backbone is None:
        raise AttributeError("Please pass backbone=<nn.Module> (e.g. model.xray_features or model.ct_backbone)")

    # ensure tensors are on device and have batch dim
    def ensure_tensor(t, default_shape=(1, 1, 224, 224), dtype=torch.float32):
        if t is None:
            return torch.zeros(default_shape, dtype=dtype, device=device)
        return t.to(device)

    xray_tensor = ensure_tensor(xray_tensor)    # (1, C, H, W)
    ct_tensor = ensure_tensor(ct_tensor)
    if input_ids is None:
        input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        input_ids = input_ids.to(device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    feats = None
    grads = None

    def fwd_hook(module, inp, out):
        nonlocal feats
        try:
            # clone & detach immediately to avoid view+inplace problems later
            feats = out.detach().clone()
            print(f"[gradcam] fwd_hook: out type={type(out)}, shape={getattr(out,'shape',None)}, dtype={out.dtype}")
        except Exception as e:
            print("[gradcam] fwd_hook exception:", e)
            feats = None

    def bwd_hook(module, grad_in, grad_out):
        nonlocal grads
        try:
            grads_tensor = grad_out[0] if isinstance(grad_out, tuple) else grad_out
            # clone & detach to avoid view/inplace problems
            grads = grads_tensor.detach().clone()
            print(f"[gradcam] bwd_hook: grad shape={getattr(grads,'shape',None)}, dtype={grads.dtype}")
        except Exception as e:
            print("[gradcam] bwd_hook exception:", e)
            grads = None

    # register hooks on the exact backbone module provided
    handle_fwd = backbone.register_forward_hook(fwd_hook)
    try:
        handle_bwd = backbone.register_full_backward_hook(bwd_hook)
    except AttributeError:
        handle_bwd = backbone.register_backward_hook(bwd_hook)

    # forward & backward
    model.zero_grad()

    # try calls in a safe way and print debug info if any call raises
    try:
        logits = model(xray_tensor, ct_tensor, input_ids, attention_mask)
    except Exception as e:
        handle_fwd.remove()
        handle_bwd.remove()
        print("[gradcam] Failed to call model inside make_gradcam_heatmap:", repr(e))
        raise

    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())

    loss = logits[0, target_class]
    try:
        loss.backward()
    except Exception as e:
        # remove hooks before raising so subsequent runs won't have dangling hooks
        handle_fwd.remove()
        handle_bwd.remove()
        print("[gradcam] backward failed:", repr(e))
        raise

    # remove hooks
    handle_fwd.remove()
    handle_bwd.remove()

    if feats is None:
        raise RuntimeError("Failed to capture features from the backbone (feats is None). Check backbone argument.")
    if grads is None:
        raise RuntimeError("Failed to capture grads from the backbone (grads is None). Check backward hook.")

    # debug print of captured tensors
    try:
        print(f"[gradcam] captured feats shape={feats.shape} dtype={feats.dtype}")
    except Exception:
        print("[gradcam] captured feats: unable to print shape")

    try:
        print(f"[gradcam] captured grads shape={grads.shape} dtype={grads.dtype}")
    except Exception:
        print("[gradcam] captured grads: unable to print shape")

    # feats & grads shape: (1, C, H, W)
    if grads.dim() != 4 or feats.dim() != 4:
        print("[gradcam] WARNING: grads/feats unexpected dims:",
              getattr(grads, "shape", None), getattr(feats, "shape", None))

    weights = grads.mean(dim=(2, 3), keepdim=True)        # (1,C,1,1)
    cam = (weights * feats).sum(dim=1, keepdim=True)      # (1,1,H,W)
    cam = F.relu(cam)

    cam_np = cam.squeeze().cpu().numpy()  # hopefully (H,W)

    # safety: if cam_np is scalar or unexpected shape, print debug and raise
    if cam_np.ndim != 2:
        print("[gradcam] ERROR: computed CAM is not 2D. cam_np.shape =", cam_np.shape, " dtype=", cam_np.dtype)
        # try to salvage: if it's (,) or zero-dim, return zeros of expected size
        try:
            h_w = (feats.shape[2], feats.shape[3])
            print("[gradcam] returning zero heatmap with shape", h_w)
            return np.zeros(h_w, dtype=np.float32)
        except Exception:
            raise RuntimeError("Grad-CAM produced invalid shape and cannot salvage.")

    # normalize 0..1
    cam_np -= cam_np.min()
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()
    else:
        print("[gradcam] NOTE: cam max == 0, returning zeros")

    print(f"[gradcam] final heatmap shape={cam_np.shape} min={cam_np.min():.6f} max={cam_np.max():.6f}")
    return cam_np



    # 0–1 float heatmap


def overlay_heatmap_on_image(orig_img, heatmap, out_path):
    """Overlay 0–1 heatmap on grayscale original image."""
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
