import cv2
import matplotlib.pyplot as plt
import os

def preview_image(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    # Load image (grayscale for X-ray)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Failed to load image.")
        return

    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.title("Sample X-ray Preview")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Change this filename later to your actual sample image
    sample_path = "data/sample/sample_image.jpg"

    preview_image(sample_path)
