import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("./assets/tiles.png")

# Scale down by 50% (0.5)
scale_percent = 0.15
width = int(img.shape[1] * scale_percent)
height = int(img.shape[0] * scale_percent)
dim = (width, height)

# Resize the actual image data
resized_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

canny = cv.Canny(resized_img, 125, 175)

cv.imshow("Canny edges", canny)

cv.waitKey(0)

cv.destroyAllWindows()