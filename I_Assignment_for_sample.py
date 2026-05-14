import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# -----------------------------
# PATHS
# -----------------------------
image_dir = "sample/images"
gt_dir = "sample/segmentations"
output_dir = "output"
metrics_file = "metrics.txt"

os.makedirs(output_dir, exist_ok=True)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# -----------------------------
# FUNCTIONS
# -----------------------------
def kmeans_seg(image, k=3):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    Z = lab.reshape((-1, 3)).astype(np.float32)

    _, label, center = cv2.kmeans(
        Z, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
        10, cv2.KMEANS_RANDOM_CENTERS
    )

    label = label.flatten()
    brightness = center[:, 0]  # L channel
    fg_label = np.argmax(brightness)
    mask = np.where(label == fg_label, 255, 0).astype(np.uint8)
    return mask.reshape(image.shape[:2])

def otsu_seg(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def refine(mask, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def align_mask(pred, gt):
    """Flip mask polarity if inverted relative to GT."""
    if np.sum(pred == gt) < np.sum(pred != gt):
        return 255 - pred
    return pred

def iou(a, b):
    a = (a > 127)
    b = (b > 127)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union != 0 else 0.0

def dice(a, b):
    a = (a > 127)
    b = (b > 127)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return (2 * inter) / denom if denom != 0 else 0.0

def compute_ssim(a, b):
    return ssim(a, b, data_range=255)

# -----------------------------
# PROCESS
# -----------------------------
all_km_iou, all_ot_iou = [], []
all_km_dice, all_ot_dice = [], []

with open(metrics_file, "w", encoding="utf-8") as f:

    classes = sorted(os.listdir(image_dir))[:10]
    print(f"Found {len(classes)} classes: {classes}\n")

    for cls in classes:
        print(f"Processing class: {cls}")

        img_path_cls = os.path.join(image_dir, cls)
        gt_path_cls  = os.path.join(gt_dir, cls)
        out_cls      = os.path.join(output_dir, cls)
        os.makedirs(out_cls, exist_ok=True)

        cls_km_iou, cls_ot_iou = [], []
        cls_km_dice, cls_ot_dice = [], []

        img_files = sorted([
            f for f in os.listdir(img_path_cls)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        ])
        print(f"  {len(img_files)} images found")

        for img_name in img_files:
            img_path = os.path.join(img_path_cls, img_name)
            base     = os.path.splitext(img_name)[0]
            gt_path  = os.path.join(gt_path_cls, base + ".png")

            if not os.path.exists(gt_path):
                print(f"  [SKIP] GT not found: {gt_path}")
                continue

            img = cv2.imread(img_path)
            gt  = cv2.imread(gt_path)

            if img is None:
                print(f"  [SKIP] Could not read image: {img_path}")
                continue
            if gt is None:
                print(f"  [SKIP] Could not read GT: {gt_path}")
                continue

            gt = cv2.resize(gt, (img.shape[1], img.shape[0]))

            # --- Segmentation ---
            km = kmeans_seg(img)         # returns binary mask directly
            ot = otsu_seg(img)

            # --- GT to binary ---
            gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
            _, gt_bin = cv2.threshold(gt_gray, 127, 255, cv2.THRESH_BINARY)

            # --- Align polarity ---
            km = align_mask(km, gt_bin)
            ot = align_mask(ot, gt_bin)

            # --- Refine ---
            km = refine(km)
            ot = refine(ot)

            # --- Metrics ---
            km_iou_val  = iou(km, gt_bin)
            ot_iou_val  = iou(ot, gt_bin)
            km_dice_val = dice(km, gt_bin)
            ot_dice_val = dice(ot, gt_bin)
            km_ssim_val = compute_ssim(km, gt_bin)
            ot_ssim_val = compute_ssim(ot, gt_bin)

            cls_km_iou.append(km_iou_val)
            cls_ot_iou.append(ot_iou_val)
            cls_km_dice.append(km_dice_val)
            cls_ot_dice.append(ot_dice_val)

            # --- Save masks as PNG ---
            km_out = os.path.join(out_cls, f"km_{base}.png")
            ot_out = os.path.join(out_cls, f"ot_{base}.png")

            ok_km = cv2.imwrite(km_out, km)
            ok_ot = cv2.imwrite(ot_out, ot)

            if not ok_km:
                print(f"  [WARN] Failed to save: {km_out}")
            if not ok_ot:
                print(f"  [WARN] Failed to save: {ot_out}")

            # --- Write per-image metrics ---
            f.write(f"{cls}/{img_name}\n")
            f.write(f"  KMeans : IoU={km_iou_val:.3f}  Dice={km_dice_val:.3f}  SSIM={km_ssim_val:.3f}\n")
            f.write(f"  Otsu   : IoU={ot_iou_val:.3f}  Dice={ot_dice_val:.3f}  SSIM={ot_ssim_val:.3f}\n\n")

        # --- Per-class summary ---
        if cls_km_iou:
            cm_km  = np.mean(cls_km_iou)
            cm_ot  = np.mean(cls_ot_iou)
            cd_km  = np.mean(cls_km_dice)
            cd_ot  = np.mean(cls_ot_dice)

            summary = (
                f"[{cls}] MEAN → "
                f"KMeans: IoU={cm_km:.3f} Dice={cd_km:.3f} | "
                f"Otsu: IoU={cm_ot:.3f} Dice={cd_ot:.3f}"
            )
            print(f"  {summary}")
            f.write(summary + "\n")
            f.write("-" * 60 + "\n\n")

            all_km_iou.extend(cls_km_iou)
            all_ot_iou.extend(cls_ot_iou)
            all_km_dice.extend(cls_km_dice)
            all_ot_dice.extend(cls_ot_dice)
        else:
            print(f"  [WARN] No valid images processed for class: {cls}")

    # --- Overall summary ---
    if all_km_iou:
        f.write("=" * 60 + "\n")
        f.write("OVERALL\n")
        f.write(f"  KMeans : IoU={np.mean(all_km_iou):.3f}  Dice={np.mean(all_km_dice):.3f}\n")
        f.write(f"  Otsu   : IoU={np.mean(all_ot_iou):.3f}  Dice={np.mean(all_ot_dice):.3f}\n")

        print("\n--- OVERALL ---")
        print(f"  KMeans : IoU={np.mean(all_km_iou):.3f}  Dice={np.mean(all_km_dice):.3f}")
        print(f"  Otsu   : IoU={np.mean(all_ot_iou):.3f}  Dice={np.mean(all_ot_dice):.3f}")

print("\nDone.")