import cv2 as cv
import numpy as np
from cv2.typing import MatLike, Size
from matplotlib import pyplot as plt

img: MatLike | None = cv.imread("./assets/tiles.png")
assert img is not None

# Scale down by 50% (0.5)
scale_percent = 0.15
width = int(img.shape[1] * scale_percent)
height = int(img.shape[0] * scale_percent)
dim: Size = (width, height)

# Resize the actual image data
resized_img = cv.resize(src=img, dsize=dim, interpolation=cv.INTER_AREA)

k_size = (10, 10)

blurred = cv.medianBlur(resized_img, 7)
cv.imshow("Median", blurred)
canny = cv.Canny(blurred, 125, 175)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

cv.imshow("canny edges", canny)

print(len(contours))
print(len((hierarchies[0])))

_ = cv.waitKey(0)

cv.destroyAllWindows()
