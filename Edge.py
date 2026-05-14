import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Black_Footed_Albatross_0001_796111.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


canny = cv2.Canny(gray, 100, 200)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)

laplacian = cv2.Laplacian(gray, cv2.CV_64F)

plt.figure(figsize=(10, 6))

plt.subplot(2,2,1)
plt.imshow(gray, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(canny, cmap='gray')
plt.title("Canny Edge")
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(sobel, cmap='gray')
plt.title("Sobel Edge")
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Edge")
plt.axis('off')

plt.tight_layout()
plt.show()