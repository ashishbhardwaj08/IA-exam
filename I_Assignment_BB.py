import cv2

# -----------------------------
# INPUT (example)
# -----------------------------
image_path = "Black_Footed_Albatross_0001_796111.jpg"   # path to your image

# CUB format: image_id, x, y, width, height
bbox = [1, 60, 27,325, 304]   # example values

# -----------------------------
# READ IMAGE
# -----------------------------
img = cv2.imread(image_path)

# Extract bounding box values
_, x, y, w, h = bbox

# Convert to integer (important)
x, y, w, h = int(x), int(y), int(w), int(h)

# -----------------------------
# DRAW BOUNDING BOX
# -----------------------------
top_left = (x, y)
bottom_right = (x + w, y + h)

cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

# -----------------------------
# SHOW IMAGE
# -----------------------------
cv2.imshow("Bounding Box", img)
cv2.waitKey(0)
cv2.destroyAllWindows()