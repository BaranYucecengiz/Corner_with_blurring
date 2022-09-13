from utils import *
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.measure import label
path = "projeresmi.png"
img = readandShow(path)
size = (600, 600)
img = cv.resize(img, size, interpolation=cv.INTER_AREA)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Threshold - Erosion - EdgeDetection
th = intensitySlice(gray, 125, 255)
kernel = np.full((3, 3), 255)
er = travel(th,erosion, (3, 3), kernel)
edge = travel(er, edgeDetection, (3, 3))
median = travel(th,medianFilter,(35,35))

# Finding the non-intersecting points using th and median
xor =np.bitwise_xor(th,median)

# Applying erosion to to get rid of unnecessary points
kernel = np.full((3, 3), 255)
er = travel(xor,erosion, (3, 3), kernel)

# Cropping to eliminate unnecessary pixels on the image
er= er[5:595, 5:595]
edge = edge[5:595, 5:595]

labels = labelling(edge)
shapes = ["Circle", "", "", "Triangle", "Rectangle", "Pentagon", "Hexagon"]
font = cv.FONT_HERSHEY_SIMPLEX
img = img[8:592, 8:592]
for i in range(len(labels)):
    r1, r2, c1, c2 =findPixelsLoc(labels[i], edge.shape, 0)
    l = er[r1: r2, c1: c2]
    lcopy = np.copy(l)
    lcopy[l ==0] = 255
    lcopy[l == 255] = 0
    #plt.imshow(lcopy)
    #plt.show()
    corners = np.unique(label(lcopy))
    cv.putText(img, shapes[max(corners)], ((c1 + c2)//2, (r1 + r2)//2), font, 0.5, (0, 0, 255), 2, cv.LINE_AA)

plt.imshow(img)
plt.show()
