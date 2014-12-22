#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import cv2
import numpy as np

showcount = 0


def load_raw(fn, mode):
	#Raw画像の読み込み
	# バイナリモードでファイル読み込み
	f = open(fn, "rb")
	# 8bitのRAW画像としてデータ取得
	im = Image.fromstring(mode, (256, 256), f.read())
	# pilからopencvにフォーマット変換して返す
	return np.asarray(im)


def block_out(matrix, size, block_list):
	#行列分解
	# 8 * 8 =　64　のウィンドウの行列に分解

	if size % 2 == 1:
		print "plz put even . not odd"
		return 0

	block_list1 = []

	block_list1 = np.vsplit(matrix, int(size))

	for i in range(0, int(size)):

		block_list2 = np.hsplit(block_list1[i], int(size))

		for x in range(0, int(size)):
			show(block_list2[x], 2)
			block_list.append(block_list2[x])


def block_in(matrix, size, block_list):
	#行列結合

	if size % 2 == 1:
		print "plz put even . not odd"
		return 0

	list1 = []
	#list1に各行の先頭ブロックを追加
	for i, (tmp) in enumerate(block_list):
		if i % size == 0:
			list1.append(tmp)

	tmp = np.array(0)

	for i in range(0, size * size - 1):
		if i % size != 0:
			list1[int(i / size)] = np.hstack((list1[int(i / size)], block_list[i + 1]))

	matrix = list1[0]
	for i in range(1, size - 1):
		matrix = np.vstack((matrix, list1[i]))

	show(matrix, 0)


def show(src, which):
#デバック用

	global showcount

	if which == 1:
		cv2.imshow('window-' + str(showcount), src)
		cv2.waitKey(0)
	elif which == 2:
		pass
	else:
		cv2.imshow('window-' + str(showcount), src)

	showcount += 1


def dither(block_list, new_block_list):
	#4*4の行列を想定 画素値をパターンに置き換える

	src = np.array([[16, 8, 14, 6], [4, 12, 2, 10], [13, 5, 15, 7], [1, 9, 3, 11]])

	for tmp in block_list:
		tmp_src = np.copy(tmp)
		tmp_src[src >= int(np.mean(tmp) / 17)] = 0
		tmp_src[src < int(np.mean(tmp) / 17)] = 255

		new_block_list.append(tmp_src)

	print src


def dither_err(matrix):

	imgArray = matrix.copy()
	outArray = np.copy(imgArray)

	maxcol = matrix.shape[0]
	maxrow = matrix.shape[1]

	for j in range(maxcol):
		for i in range(maxrow):

			if(i != maxrow - 1)and(j != maxcol - 1):

				if imgArray[i, j] > 128:

					outArray[i, j] = 255
					error = imgArray[i, j] - 255
			
					imgArray[i + 1, j] = int(imgArray[i + 1, j] + (7 / 16.0) * error)
					imgArray[i - 1, j + 1] = int(imgArray[i - 1, j + 1] + (3 / 16.0) * error)
					imgArray[i, j + 1] = int(imgArray[i, j + 1] + (5 / 16.0) * error)
					imgArray[i + 1, j + 1] = int(imgArray[i + 1, j + 1] + (1 / 16.0) * error)
					

				if imgArray[i, j] < 128:

					outArray[i, j] = 0
					error = imgArray[i, j] - 0
					imgArray[i + 1, j] = imgArray[i + 1, j] + (7 / 16.0) * error
					imgArray[i - 1, j + 1] = imgArray[i - 1, j + 1] + (3 / 16.0) * error
					imgArray[i, j + 1] = imgArray[i, j + 1] + (5 / 16.0) * error
					imgArray[i + 1, j + 1] = imgArray[i + 1, j + 1] + (1 / 16.0) * error

	return outArray


def dither_errP(matrix):

	tmpArray = matrix.copy()
	outArray = np.copy(tmpArray)
	#Numpy to pil
	imgArray = Image.fromarray(np.uint8(tmpArray))
	print imgArray

	for y in range(matrix.shape[1]):
		for x in range(1, matrix.shape[0]):

			if(x != (matrix.shape[0] - 1))and(y != (matrix.shape[1] - 1)):
		
				if imgArray.getpixel((x, y)) > 128:
					outArray[y, x] = 255
					error = imgArray.getpixel((x, y)) - 255

					imgArray.putpixel((x + 1, y), int(imgArray.getpixel((x + 1, y)) + (7 / 16.0) * error))
					imgArray.putpixel((x - 1, y + 1), int(imgArray.getpixel((x - 1, y + 1)) + (3 / 16.0) * error))
					imgArray.putpixel((x, y + 1), int(imgArray.getpixel((x, y + 1)) + (5 / 16.0) * error))
					imgArray.putpixel((x + 1, y + 1), int(imgArray.getpixel((x + 1, y + 1)) + (1 / 16.0) * error))

				if imgArray.getpixel((x, y)) < 128:
					outArray[y, x] = 0
					error = imgArray.getpixel((x, y)) - 0
					
					imgArray.putpixel((x + 1, y), int(imgArray.getpixel((x + 1, y)) + (7 / 16.0) * error))
					imgArray.putpixel((x - 1, y + 1), int(imgArray.getpixel((x - 1, y + 1)) + (3 / 16.0) * error))
					imgArray.putpixel((x, y + 1), int(imgArray.getpixel((x, y + 1)) + (5 / 16.0) * error))
					imgArray.putpixel((x + 1, y + 1), int(imgArray.getpixel((x + 1, y + 1)) + (1 / 16.0) * error))

	return outArray


if __name__ == '__main__':

	im = load_raw('/Users/keiichi/Downloads/Files/lenna.256', "L")

	print im

	length = im.shape[0]

	print "length = ", length

	block_list = []
	new_block_list = []
	block_out(im, length / 4, block_list)

	imout = np.copy(im)
	dither(block_list, new_block_list)
	block_in(imout, length / 4, new_block_list)

	res = dither_err(im)
	show(res, 0)

	res2 = dither_errP(im)
	show(res2, 0)

	# 画像表示
	cv2.imshow("Show Raw Image", im)
	# キー入力待機
	cv2.waitKey(0)
	# ウィンドウ破棄
	cv2.destroyAllWindows()
