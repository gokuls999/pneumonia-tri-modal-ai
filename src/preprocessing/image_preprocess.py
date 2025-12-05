import cv2
import numpy as np
import torch
import os

# Config
TARGET_SIZE = (224, 224)  # H, W

def denoise(img):
    # fast non-local means denoising (works well for X-rays)
    return cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)

def clahe_enhance(img):
    # img expected single-channel uint8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def zscore_normalize(img_float):
    # img_float: float32
    mean = img_float.mean()
    std = img_float.std() if img_float.std() > 0 else 1.0
    return (img_float - mean) / std

def resize_image(img, size=TARGET_SIZE):
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)

def preprocess_image_file(image_path, out_tensor_path=None, to_tensor=True):
    """
    Reads a grayscale image, denoises, enhances, normalizes, resizes and optionally saves a torch tensor.
    Returns: numpy array (H,W) normalized float32 or torch.Tensor (1,H,W) if to_tensor=True
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # load grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Failed to read image")

    # preprocessing pipeline
    img = denoise(img)
    img = clahe_enhance(img)
    img = resize_image(img, TARGET_SIZE)
    img = img.astype(np.float32)
    img = zscore_normalize(img)

    if to_tensor:
        # convert to (1, H, W) torch tensor
        tensor = torch.from_numpy(img).unsqueeze(0)
        if out_tensor_path:
            torch.save(tensor, out_tensor_path)
        return tensor
    else:
        if out_tensor_path:
            # save numpy
            np.save(out_tensor_path, img)
        return img

if __name__ == "__main__":
    sample_in = "data/sample/sample_image.jpg"
    sample_out = "data/processed/sample_image.pt"
    try:
        t = preprocess_image_file(sample_in, sample_out)
        print("Preprocessing done. Saved to:", sample_out)
        print("Tensor shape:", tuple(t.shape))
    except Exception as e:
        print("Error:", e)
