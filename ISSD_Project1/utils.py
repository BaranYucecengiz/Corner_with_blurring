import numpy as np
import cv2 as cv
from skimage.measure import label
from scipy import signal as sig
import scipy.ndimage as ndi


def padding(img, filterR, filterC):
	return np.pad(  array=img,
					pad_width=((filterR // 2, filterR // 2), (filterC // 2, filterC // 2)),
	                mode='constant',
	                constant_values=(0, 0))


def readandShow(filename):
	img = cv.imread(filename)
	# cv.imshow('', img)
	# cv.waitKey(0)
	# cv.destroyAllWindows()
	return img


def edgeDetection(loc, fr, fc):
	value = (int(loc[fr // 2][0]) + int(loc[0][fr // 2]) \
	         + int(loc[fr - 1][fc // 2]) \
	         + int(loc[fr // 2][fc - 1])) \
	        - (4 * int(loc[fr // 2][fc // 2]))
	return value


def medianFilter(loc):
	return np.median(loc)


def gaussKernel(R, C, standartdeviation, K):
	final = np.zeros((R, C))

	for r in range(R):
		indexRow = r - R // 2
		for c in range(C):
			indexColumn = c - C // 2
			# Gauss Formula
			up = indexRow ** 2 + indexColumn ** 2
			down = 2 * (standartdeviation ** 2)
			gauss = K * np.exp(-(up / down))
			final[r, c] = gauss
	return final / final.sum()  # Low- pass filter


def boxFilter(R, C, value):
	box = np.full((R, C), value)
	return box / box.sum()


def corr(loc, kernel):
	return (loc * kernel).sum()


def conv(loc, kernel):
	kernel = np.fliplr(kernel)
	kernel = np.flipud(kernel)
	return corr(loc, kernel)


def intensitySlice(img, minVal, maxVal):
	copy = np.array(img).copy()
	copy[(copy > minVal) & (copy < maxVal)] = 255
	copy[(copy < minVal) | (copy > maxVal)] = 0

	return copy


def labelling(img):
	"""
	classify all decoupled pixels
	"""
	listshapes = list()
	arr = label(np.array(img))
	for i in range(len(np.unique(arr)) - 1):
		copy = np.array(arr).copy()
		copy[arr == i + 1] = 0
		copy[arr != i + 1] = 255
		listshapes.append(copy)

	return listshapes


def dilation(loc, kernel):
	if len(np.intersect1d(loc, kernel)) == 0:
		return 255
	else:
		return 0


def erosion(loc, kernel):
	if np.array_equal(loc, kernel):
		return 0
	else:
		return 255


def travel(img, operation, kernelSIZE, kernel=None):
	imgRows, imgColumns = np.array(img).shape[:2]
	filterRows, filterColumns = kernelSIZE

	padImg = padding(img, filterRows, filterColumns)
	finalImg = np.zeros_like(img)

	for row in range(imgRows):
		for column in range(imgColumns):
			# print(f"Row : {row}, Column {column}")
			loc = padImg[row: (row + filterRows), column: (column + filterColumns)]

			if operation == edgeDetection:
				value = operation(loc, filterRows, filterColumns)
				if value > 0:
					finalImg[row, column] = 255
				else:
					finalImg[row, column] = 0
			elif operation == medianFilter:
				finalImg[row, column] = operation(loc)
			else:
				finalImg[row, column] = operation(loc, kernel)

	return finalImg


def compress0to255(src):
	src = src.astype(float)
	src = np.absolute(src)
	src -= np.min(src)
	src /= np.max(src)
	src = 255 * src
	src = src.astype(np.uint8)
	return src


def findPixelsLoc(img, size, value):
	indicesRow = list()
	indicesColumn = list()

	rows = size[0]
	columns = size[1]
	# print(rows, columns)
	for row in range(rows):
		for column in range(columns):
			if img[row, column] == value:
				indicesRow.append(row)
				indicesColumn.append(column)

	r1 = min(indicesRow)
	r2 = max(indicesRow)
	c1 = min(indicesColumn)
	c2 = max(indicesColumn)
	return r1 - 5, r2 + 5, c1 - 5, c2 + 5


def DFS(src, visits, row, col, cluster):
	visits[row][col] = 255
	n = neighbors(row, col)
	for i, j in n:
		if src[i][j] == 255 and visits[i][j] == 0:
			cluster.append((i, j))
			DFS(src, visits, i, j, cluster)


def neighbors(row, col):
	n = list()
	for i in range(row - 1, row + 2):
		for j in range(col - 1, col + 2):
			n.append((i, j))
	return n


def findAllPixels(src):
	visits = np.zeros(src.shape, dtype=np.uint8)
	cluster = list()

	rows, cols = src.shape
	for row in range(1, rows - 1):
		for col in range(1, cols - 1):
			if src[row][col] == 255 and visits[row][col] == 0:
				# print(row, col)
				DFS(src, visits, row, col, cluster)

	return cluster


def isLineer(p1, p2, p3):
	import math
	x1, y1 = p1
	x2, y2 = p2
	x3, y3 = p3

	try:
		slope = (y2 - y1) / (x2 - x1)
	except ZeroDivisionError:
		slope = 0
		return math.isclose(x3 - x1, slope * (y3 - y1), rel_tol=9e-01)
	return math.isclose(y3 - y1, slope * (x3 - x1), rel_tol=9e-01)


# import skimage
# def noise(img):
#
# 	out = skimage.util.random_noise(img, mode="gaussian")
# 	norm_image = cv.normalize(out, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
#
# 	return norm_image.astype(np.uint8)

def gradient_x(gray):
	##Sobel operator kernels.
	kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	return sig.convolve2d(gray, kernel_x, mode='same')


def gradient_y(gray):
	kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	return sig.convolve2d(gray, kernel_y, mode='same')


def calcResponse(src, val, k):
	I_x = gradient_x(src)
	I_y = gradient_y(src)

	Ixx = I_x ** 2
	Ixy = I_y * I_x
	Iyy = I_y ** 2

	Ixx = ndi.gaussian_filter(Ixx, sigma=1)
	Ixy = ndi.gaussian_filter(Ixy, sigma=1)
	Iyy = ndi.gaussian_filter(Iyy, sigma=1)

	# determinant
	detA = Ixx * Iyy - Ixy ** 2
	# trace
	traceA = Ixx + Iyy
	harris_response = detA - k * traceA ** 2

	copy = np.copy(harris_response)
	harris_response = harris_response * (1 / 10 ** 8)

	copy[harris_response >= val] = 255
	copy[harris_response <= val] = 0

	return copy
