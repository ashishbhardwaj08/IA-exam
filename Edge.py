import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("Black_Footed_Albatross_0001_796111.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------------- EDGE DETECTORS ---------------- #

# 1. Canny Edge Detector
canny = cv2.Canny(gray, 100, 200)

# 2. Sobel Edge Detector
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)

# 3. Laplacian Edge Detector
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# 4. Prewitt Edge Detector
kernelx = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]])

kernely = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])

prewittx = cv2.filter2D(gray, -1, kernelx)
prewitty = cv2.filter2D(gray, -1, kernely)
prewitt = cv2.magnitude(prewittx.astype(np.float64),
                        prewitty.astype(np.float64))

# 5. Roberts Edge Detector
kernelx_roberts = np.array([[1, 0],
                            [0, -1]])

kernely_roberts = np.array([[0, 1],
                            [-1, 0]])

robertsx = cv2.filter2D(gray, -1, kernelx_roberts)
robertsy = cv2.filter2D(gray, -1, kernely_roberts)

roberts = cv2.magnitude(robertsx.astype(np.float64),
                        robertsy.astype(np.float64))

# 6. Scharr Edge Detector
# scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
# scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
# scharr = cv2.magnitude(scharrx, scharry)

# ---------------- DISPLAY RESULTS ---------------- #

plt.figure(figsize=(15, 10))

plt.subplot(2,3,1)
plt.imshow(gray, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(canny, cmap='gray')
plt.title("Canny")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(sobel, cmap='gray')
plt.title("Sobel")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(prewitt, cmap='gray')
plt.title("Prewitt")
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(roberts, cmap='gray')
plt.title("Roberts")
plt.axis('off')

# plt.subplot(2,4,7)
# plt.imshow(scharr, cmap='gray')
# plt.title("Scharr")
# plt.axis('off')

plt.tight_layout()
plt.show()