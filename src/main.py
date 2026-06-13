import cv2 as cv
import numpy as np
from cv2.typing import MatLike, Size
from matplotlib import pyplot as plt

img: MatLike | None = cv.imread("../assets/tiles.png")
assert img is not None

# Scale down by 50% (0.5)
scale_percent = 0.15
width = int(img.shape[1] * scale_percent)
height = int(img.shape[0] * scale_percent)
dim: Size = (width, height)

# Resize the actual image data
resized_img = cv.resize(src=img, dsize=dim, interpolation=cv.INTER_AREA)

k_size = (10, 10)

blurred = cv.medianBlur(resized_img, 11)
cv.imshow("Median", blurred)
canny = cv.Canny(blurred, 125, 175)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

cv.imshow("canny edges", canny)

print(len(contours))

squares = []
for i, contour in enumerate(contours):
    estimated = cv.approxPolyDP(contour, epsilon=0.01 * cv.arcLength(contour, True), closed=True)
    print(f"Contour {i}: {len(estimated)} vertices, area={cv.contourArea(estimated)}")
    if len(estimated) >= 4 and cv.isContourConvex(estimated) and cv.contourArea(estimated) > 100:
        squares.append(contour)

canny_color = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
squaresImg = cv.polylines(canny_color.copy(), squares, isClosed=True, color=(0, 255, 0), thickness=2)
overlay = cv.bitwise_and(canny_color, squaresImg)
cv.imshow("Squares", overlay)
print(len((hierarchies[0])))
print(f"Found {len(squares)} squares")
_ = cv.waitKey(0)

cv.destroyAllWindows()
