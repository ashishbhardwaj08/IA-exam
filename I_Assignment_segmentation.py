import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

  

# -----------------------------
# 1. LOAD IMAGES
# -----------------------------
img_path = "Black_Footed_Albatross_0001_796111.jpg"
gt_path = "Black_Footed_Albatross_0001_796111.png"

img = cv2.imread(img_path)
gt = cv2.imread(gt_path)

if img is None:
    raise ValueError("Original image not loaded. Check path.")
if gt is None:
    raise ValueError("Ground truth image not loaded. Check path.")

# Resize GT
gt = cv2.resize(gt, (img.shape[1], img.shape[0]))

# -----------------------------
# 2. SEGMENTATION METHODS
# -----------------------------

# K-Means
def kmeans_segmentation(image, k=3):
    pixel_vals = image.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    _, labels, centers = cv2.kmeans(
        pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(image.shape)
    return segmented


# Otsu
def otsu_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# -----------------------------
# 3. APPLY SEGMENTATION
# -----------------------------
kmeans_img = kmeans_segmentation(img)
otsu_img = otsu_segmentation(img)

# -----------------------------
# 4. CONVERT TO BINARY MASKS
# -----------------------------

# Ground Truth → binary
gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
_, gt_bin = cv2.threshold(gt_gray, 127, 255, cv2.THRESH_BINARY)
gt_bin = gt_bin / 255.0

# K-Means → binary using Otsu
kmeans_gray = cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2GRAY)
_, kmeans_bin = cv2.threshold(
    kmeans_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
kmeans_bin = kmeans_bin / 255.0

# Otsu already binary
otsu_bin = otsu_img / 255.0

# Closing
# kmeans_bin = cv2.morphologyEx((kmeans_bin*255).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8)) / 255.0
# otsu_bin = cv2.morphologyEx((otsu_bin*255).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8)) / 255.0


# -----------------------------
# 5. ALIGN FOREGROUND (AUTO FIX)
# -----------------------------
def align_mask(pred, gt):
    intersection = np.logical_and(pred > 0.5, gt > 0.5).sum()
    if intersection < 0.1 * (pred > 0.5).sum():
        print("Inverting mask for alignment...")
        pred = 1 - pred
    return pred

kmeans_bin = align_mask(kmeans_bin, gt_bin)
otsu_bin = align_mask(otsu_bin, gt_bin)

# -----------------------------
# 6. DISTANCE METRICS
# -----------------------------

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def mse(a, b):
    return np.mean((a - b) ** 2)

def compute_ssim(a, b):
    score, _ = ssim(a, b, full=True, data_range=1.0)
    return score

def compute_iou(a, b):
    a = (a > 0.5).astype(np.uint8)
    b = (b > 0.5).astype(np.uint8)

    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    return intersection / union if union != 0 else 0

# -----------------------------
# 7. RESULTS
# -----------------------------
print("----- K-MEANS vs GROUND TRUTH -----")
print("Euclidean:", euclidean_distance(kmeans_bin, gt_bin))
print("MSE:", mse(kmeans_bin, gt_bin))
print("SSIM:", compute_ssim(kmeans_bin, gt_bin))
print("IoU:", compute_iou(kmeans_bin, gt_bin))

print("\n----- OTSU vs GROUND TRUTH -----")
print("Euclidean:", euclidean_distance(otsu_bin, gt_bin))
print("MSE:", mse(otsu_bin, gt_bin))
print("SSIM:", compute_ssim(otsu_bin, gt_bin))
print("IoU:", compute_iou(otsu_bin, gt_bin))

# -----------------------------
# 8. DISPLAY
# -----------------------------
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gt_bin, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(kmeans_bin, cmap="gray")
plt.title("K-Means Binary")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(otsu_bin, cmap="gray")
plt.title("Otsu Binary")
plt.axis("off")

plt.tight_layout()
plt.show()