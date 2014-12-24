#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
created by keiichi 
"""

from PIL import Image
import cv2
import numpy as np

#/Users/keiichi/Downloads/Files/lenna.256
#data = np.loadtxt('/Users/keiichi/Downloads/Files/lenna.256', comments='#' ,delimiter=',')
#print data

showcount = 0


def load_raw(fn, mode, size):
	#Raw画像の読み込み
	# バイナリモードでファイル読み込み
	f = open(fn, "rb")
	# 8bitのRAW画像としてデータ取得
	print type(f)
	#im = Image.fromstring(mode, (256, 256), f.read())
	im = Image.fromstring(mode, (size, size), f.read())
	#im.show()
	# pilからopencvにフォーマット変換して返す
	return np.asarray(im)


def block_out(matrix, size, block_list):
	#行列分解

	# 8 * 8 =　64　のウィンドウの行列に分解
	#block_list = []

	if size % 2 == 1:
		print "plz put even . not odd"
		return 0

	block_list1 = []

	#for i in range(0, 8):
	block_list1 = np.vsplit(matrix, int(size))

	#print block_list1[0].shape

	for i in range(0, int(size)):
		#print np.hsplit(block_list1[i],8)
		block_list2 = np.hsplit(block_list1[i], int(size))

		for x in range(0, int(size)):
			show(block_list2[x], 2)
			block_list.append(block_list2[x])

		#block_list.append(block_list2)
		#block_list.append(np.hsplit(block_list1[i],8))

	#print len(block_list)
	#print block_list


	#src_list = block_list

	#print block_list[1].shape


	#return


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

	#block_count = 0
	tmp = np.array(0)

	for i in range(0, size * size - 1):
		if i % size != 0:
			list1[int(i / size)] = np.hstack((list1[int(i / size)], block_list[i + 1]))

	matrix = list1[0]
	for i in range(1, size - 1):
		matrix = np.vstack((matrix, list1[i]))
		#list1[int(block_count/8)]=

	#for tmp in list1:
	#	show(tmp,0)
	#print "tmp",tmp
	show(matrix, 0)

	return matrix


def show(src, which):
#デバック用

	global showcount

	if which == 1:
		cv2.imshow('window-' + str(showcount), src)
		cv2.waitKey(0)
	elif which == 2:
		pass
		#print "Dontshow"

	else:
		cv2.imshow('window-' + str(showcount), src)

	showcount += 1

	#4*4のブロック１つ１つに対して、８方向の勾配ヒストグラムの作成
	#4*4*8 = 128
	#
	#************/
	#************/


def dither(block_list, new_block_list):
	#4*4の行列を想定 画素値をパターンに置き換える

	#src = np.array([[15,7,13,5],[3,11,1,9],[12,4,14,6],[0,8,2,10]])
	src = np.array([[16, 8, 14, 6], [4, 12, 2, 10], [13, 5, 15, 7], [1, 9, 3, 11]])

	for tmp in block_list:
		tmp_src = np.copy(tmp)
		tmp_src[src >= int(np.mean(tmp) / 17)] = 0
		tmp_src[src < int(np.mean(tmp) / 17)] = 255

		#tmp_src[src>=int(np.mean(tmp)/16)]=0
		#tmp_src[src<int(np.mean(tmp)/16)]=255

		new_block_list.append(tmp_src)

		#show(tmp_src,0)

	print src


def dither_err(matrix):

	#kernel = np.array([0,0,0],[0,0,7],[3,5,1])/16

	#res = np.copy(matrix)
	#res = scipy.signal.convolve2d(matrix, kernel, 'valid', 'fill')

	#return res
	#imgArray = np.copy(matrix)
	imgArray = matrix.copy()
	outArray = np.copy(imgArray)

	#print matrix.size

	#maxcol , maxrow = matrix.size #get size
	maxcol = matrix.shape[0]
	maxrow = matrix.shape[1]

	#print maxrow
	#print maxcol

	#print type(imgArray)

	#print "before",imgArray[:,0]
	#matrix.flags.writeable = True
	#imgArray.flags.writeable = True
	for j in range(maxcol):
		for i in range(maxrow):
		#for j in range(maxcol):

			if(i != maxrow - 1)and(j != maxcol - 1):

				if imgArray[i, j] > 128:

					outArray[i, j] = 255
					#print "b,",imgArray[i+1,j]
					error = imgArray[i, j] - 255
					#print error
					#print float((7/16.0)*error)
					imgArray[i + 1, j] = int(imgArray[i + 1, j] + (7 / 16.0) * error)
					imgArray[i - 1, j + 1] = int(imgArray[i - 1, j + 1] + (3 / 16.0) * error)
					imgArray[i, j + 1] = int(imgArray[i, j + 1] + (5 / 16.0) * error)
					imgArray[i + 1, j + 1] = int(imgArray[i + 1, j + 1] + (1 / 16.0) * error)
					#print "a,",imgArray[i+1,j]

				if imgArray[i, j] < 128:

					outArray[i, j] = 0
					error = imgArray[i, j] - 0
					imgArray[i + 1, j] = imgArray[i + 1, j] + (7 / 16.0) * error
					imgArray[i - 1, j + 1] = imgArray[i - 1, j + 1] + (3 / 16.0) * error
					imgArray[i, j + 1] = imgArray[i, j + 1] + (5 / 16.0) * error
					imgArray[i + 1, j + 1] = imgArray[i + 1, j + 1] + (1 / 16.0) * error

			#imgArray[i,j][0] = 255-imgArray[i,j][0] #R
			#imgArray[i,j][1] = 255-imgArray[i,j][1] #G
			#imgArray[i,j][2] = 255-imgArray[i,j][2] #B

	#print "after",imgArray[:,0]
	#print outArray[:,0]
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
			#print imgArray.getpixel((x,y))
			#print imgArray[x,y]
				if imgArray.getpixel((x, y)) > 128:
					outArray[y, x] = 255
					error = imgArray.getpixel((x, y)) - 255
					#imgArray.putpixel((x,y),255)

					imgArray.putpixel((x + 1, y), int(imgArray.getpixel((x + 1, y)) + (7 / 16.0) * error))
					imgArray.putpixel((x - 1, y + 1), int(imgArray.getpixel((x - 1, y + 1)) + (3 / 16.0) * error))
					imgArray.putpixel((x, y + 1), int(imgArray.getpixel((x, y + 1)) + (5 / 16.0) * error))
					imgArray.putpixel((x + 1, y + 1), int(imgArray.getpixel((x + 1, y + 1)) + (1 / 16.0) * error))

				if imgArray.getpixel((x, y)) < 128:
					outArray[y, x] = 0
					error = imgArray.getpixel((x, y)) - 0
					#imgArray.putpixel((x,y),0)

					imgArray.putpixel((x + 1, y), int(imgArray.getpixel((x + 1, y)) + (7 / 16.0) * error))
					imgArray.putpixel((x - 1, y + 1), int(imgArray.getpixel((x - 1, y + 1)) + (3 / 16.0) * error))
					imgArray.putpixel((x, y + 1), int(imgArray.getpixel((x, y + 1)) + (5 / 16.0) * error))
					imgArray.putpixel((x + 1, y + 1), int(imgArray.getpixel((x + 1, y + 1)) + (1 / 16.0) * error))

	#imgArray.show()
	#matrix = outArray
	return outArray


if __name__ == '__main__':

	im = load_raw('./Files/lenna.256', "L", 256)

	print im

	length = im.shape[0]

	print "length = ", length

	block_list = []
	new_block_list = []
	block_out(im, length / 4, block_list)

	#block_out(im, 2, block_list)


	count = 0
	for x in block_list:
		#show(x,0)
		#print count
		count += 1
	#cv2.waitKey(0)

	imout = np.copy(im)
	dither(block_list, new_block_list)
	imout = block_in(imout, length / 4, new_block_list)

	show(imout, 0)
	#res = dither_err(im)
	#show(res,0)

	res2 = dither_errP(im)

	show(res2, 0)

	

	#imR = load_raw('./Files/hatgirl.red', "L", 512)
	#imG = load_raw('./Files/hatgirl.grn', "L", 512)
	#imB = load_raw('./Files/hatgirl.blu', "L", 512)

	imR = load_raw('./Files/baboon.red', "L", 512)
	imG = load_raw('./Files/baboon.grn', "L", 512)
	imB = load_raw('./Files/baboon.blu', "L", 512)

	print imR.shape
	print imG.shape
	print imB.shape
	
	org = np.vstack((imB, imG))
	org = np.vstack((org, imR))

	print org.shape

	length2 = imR.shape[0]
	block_listR = []
	new_block_listR = []
	block_listG = []
	new_block_listG = []
	block_listB = []
	new_block_listB = []

	block_out(imR, length2 / 4, block_listR)
	block_out(imG, length2 / 4, block_listG)
	block_out(imB, length2 / 4, block_listB)

	imoutR = np.copy(imR)
	imoutG = np.copy(imG)
	imoutB = np.copy(imB)

	dither(block_listR, new_block_listR)
	dither(block_listG, new_block_listG)
	dither(block_listB, new_block_listB)

	imoutR = block_in(imoutR, length2 / 4, new_block_listR)
	imoutG = block_in(imoutG, length2 / 4, new_block_listG)
	imoutB = block_in(imoutB, length2 / 4, new_block_listB)

	tmp = np.dstack((imoutR, imoutG))
	tmp = np.dstack((tmp, imoutB))
	print tmp.shape
	show(tmp, 0)

	tmpim = Image.fromarray(tmp)
	tmpim.show()

	imRr = dither_errP(imR)
	imGr = dither_errP(imG)
	imBr = dither_errP(imB)

	tmp2 = np.dstack((imRr, imGr))
	tmp2 = np.dstack((tmp2, imBr))
	print tmp2.shape
	show(tmp2, 0)

	tmpim2 = Image.fromarray(tmp2)
	tmpim2.show()

	# 画像表示
	cv2.imshow("Show Raw Image", im)
	cv2.imshow("Show Raw Image2", org)
	# キー入力待機
	cv2.waitKey(0)
	# ウィンドウ破棄
	cv2.destroyAllWindows()
