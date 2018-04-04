import tensorflow as tf
import numpy as np
import cv2

def coarseness(image):
	kmax = tf.constant(5)
	image = tf.reduce_mean(image,axis=3)
	image = tf.expand_dims(image,-1)

	window1 = np.power(2,1)
	kernel1 = tf.ones([window1,window1,1,1])
	average_gray1 = tf.nn.conv2d(image,kernel1,strides = [1,1,1,1],padding='SAME')
	kernel_h1 = np.zeros([1,2*window1,1,1])
	kernel_h1[0][0][0][0] = -1
	kernel_h1[0][2*window1-1][0][0] = 1
	horizon1 = tf.nn.conv2d(average_gray1,kernel_h1,strides=[1,1,1,1],padding='SAME')
	horizon1 = tf.squeeze(horizon1, [3])
	kernel_v1 = np.zeros([2*window1,1,1,1])
	kernel_v1[0][0][0][0] = -1
	kernel_v1[2*window1-1][0][0][0] = 1
	vertical1 = tf.nn.conv2d(average_gray1,kernel_v1,strides=[1,1,1,1],padding='SAME')
	vertical1 = tf.squeeze(vertical1,[3])

	window2 = np.power(2,2)
	kernel2 = tf.ones([window2,window2,1,1])
	average_gray2 = tf.nn.conv2d(image,kernel2,strides = [1,1,1,1],padding='SAME')
	kernel_h2 = np.zeros([1,2*window2,1,1])
	kernel_h2[0][0][0][0] = -1
	kernel_h2[0][2*window2-1][0][0] = 1
	horizon2 = tf.nn.conv2d(average_gray2,kernel_h2,strides=[1,1,1,1],padding='SAME')
	horizon2 = tf.squeeze(horizon2, [3])
	kernel_v2 = np.zeros([2*window2,1,1,1])
	kernel_v2[0][0][0][0] = -1
	kernel_v2[2*window2-1][0][0][0] = 1
	vertical2 = tf.nn.conv2d(average_gray2,kernel_v2,strides=[1,1,1,1],padding='SAME')
	vertical2 = tf.squeeze(vertical2,[3])

	window3 = np.power(2,3)
	kernel3 = tf.ones([window3,window3,1,1])
	average_gray3 = tf.nn.conv2d(image,kernel3,strides = [1,1,1,1],padding='SAME')
	kernel_h3 = np.zeros([1,2*window3,1,1])
	kernel_h3[0][0][0][0] = -1
	kernel_h3[0][2*window3-1][0][0] = 1
	horizon3 = tf.nn.conv2d(average_gray3,kernel_h3,strides=[1,1,1,1],padding='SAME')
	horizon3 = tf.squeeze(horizon3, [3])
	kernel_v3 = np.zeros([2*window3,1,1,1])
	kernel_v3[0][0][0][0] = -1
	kernel_v3[2*window3-1][0][0][0] = 1
	vertical3 = tf.nn.conv2d(average_gray3,kernel_v3,strides=[1,1,1,1],padding='SAME')
	vertical3 = tf.squeeze(vertical3,[3])

	window4 = np.power(2,4)
	kernel4 = tf.ones([window4,window4,1,1])
	average_gray4 = tf.nn.conv2d(image,kernel4,strides = [1,1,1,1],padding='SAME')
	kernel_h4 = np.zeros([1,2*window4,1,1])
	kernel_h4[0][0][0][0] = -1
	kernel_h4[0][2*window4-1][0][0] = 1
	horizon4 = tf.nn.conv2d(average_gray4,kernel_h4,strides=[1,1,1,1],padding='SAME')
	horizon4 = tf.squeeze(horizon4, [3])
	kernel_v4 = np.zeros([2*window4,1,1,1])
	kernel_v4[0][0][0][0] = -1
	kernel_v4[2*window4-1][0][0][0] = 1
	vertical4 = tf.nn.conv2d(average_gray4,kernel_v4,strides=[1,1,1,1],padding='SAME')
	vertical4 = tf.squeeze(vertical4,[3])

	window5 = np.power(2,5)
	kernel5 = tf.ones([window5,window5,1,1])
	average_gray5 = tf.nn.conv2d(image,kernel5,strides = [1,1,1,1],padding='SAME')
	kernel_h5 = np.zeros([1,2*window5,1,1])
	kernel_h5[0][0][0][0] = -1
	kernel_h5[0][2*window5-1][0][0] = 1
	horizon5 = tf.nn.conv2d(average_gray5,kernel_h5,strides=[1,1,1,1],padding='SAME')
	horizon5 = tf.squeeze(horizon5, [3])
	kernel_v5 = np.zeros([2*window5,1,1,1])
	kernel_v5[0][0][0][0] = -1
	kernel_v5[2*window5-1][0][0][0] = 1
	vertical5 = tf.nn.conv2d(average_gray5,kernel_v5,strides=[1,1,1,1],padding='SAME')
	vertical5 = tf.squeeze(vertical5,[3])

	#return tf.shape(horizon5)
	horizon = tf.concat([horizon1,horizon2,horizon3,horizon4,horizon5],0)
	vertical = tf.concat([vertical1,vertical2,vertical3,vertical4,vertical5],0)
	h_max_index = tf.to_int32(tf.argmax(horizon,0))
	v_max_index = tf.to_int32(tf.argmax(vertical,0))
	h_max = tf.reduce_max(horizon,0)
	v_max = tf.reduce_max(vertical,0)
	comp = tf.greater(h_max,v_max)
	Sbest = tf.where(comp,h_max_index,v_max_index)
	#return tf.shape(Sbest)
	Sbest = tf.to_float(tf.pow(2, Sbest))
	frcs = tf.reduce_mean(Sbest)
	return frcs

def contrast(image):
	image = tf.reshape(tf.reduce_mean(image,axis=3),[-1])
	mean = tf.reduce_mean(image)
	m4 = tf.reduce_mean(tf.pow(tf.subtract(image, mean),4))
	var = tf.reduce_mean(tf.square(image-mean))
	std = tf.pow(var,0.5)
	alpha4 = tf.div(m4,tf.pow(var,2))
	fcon = tf.div(std,tf.pow(alpha4,0.25))
	return fcon

def directionality(image):
	image = tf.expand_dims(tf.reduce_mean(image,axis=3),3)
	kernel_h = tf.expand_dims(tf.expand_dims(tf.constant([[-1,0,1],[-1,0,1],[-1,0,1]], dtype='float32'),2),3)
	kernel_v = tf.expand_dims(tf.expand_dims(tf.constant([[1,1,1],[0,0,0],[-1,-1,-1]], dtype='float32'),2),3)

	deltaH = tf.nn.conv2d(image,kernel_h,strides=[1,1,1,1],padding='SAME')
	deltaV = tf.nn.conv2d(image,kernel_v,strides=[1,1,1,1],padding='SAME')
	deltaG = tf.reshape(tf.div(tf.abs(deltaH) + tf.abs(deltaV), 2.0),[-1])

	zeroH = tf.equal(deltaH,0)
	zeroV = tf.equal(deltaV,0)
	zeroBoth = tf.logical_and(zeroH, zeroV)
	deltaH_one = tf.add(deltaH,1)
	deltaH_pi = tf.add(deltaH, np.pi)
	deltaH = tf.where(zeroH,deltaH_one,deltaH)	

	theta = tf.atan(tf.div(deltaV, deltaH)) + np.pi / 2.0
	theta = tf.where(zeroH, deltaH_pi, theta)
	theta = tf.where(zeroBoth,deltaH, theta)
	theta = tf.reshape(theta,[-1])

	n = 16
	t = 12
	cnt = 0
	for ni in range(n):
		cond1 = tf.greater_equal(deltaG,t)
		cond2 = tf.greater_equal(theta,(2*ni-1) * np.pi / (2 * n))
		cond3 = tf.less(theta,(2*ni+1) * np.pi / (2*n))
		cond = tf.logical_and(cond1, cond2)
		cond = tf.logical_and(cond3,cond)
		s = tf.reshape(tf.reduce_sum(tf.to_int32(cond)),[1])
		if (ni == 0):
			hd = s
		else:
			hd = tf.concat([hd,s],0)
	hd = tf.to_float(hd)
	hd = tf.div(hd,tf.reduce_mean(hd))
	hd_max_index = tf.to_int32(tf.argmax(hd))
	r = tf.to_int32(tf.range(0,n,1))
	rp = tf.pow(tf.subtract(r,hd_max_index),2)
	hd_x = tf.expand_dims(hd,1)
	rp_x = tf.to_float(tf.transpose(tf.expand_dims(rp,1)))
	fdir = tf.squeeze(tf.matmul(rp_x,hd_x))
	return fdir



if __name__ == '__main__':
	img = np.array(cv2.imread('stata.jpg'))[np.newaxis, ]
	with tf.Session() as sess:
		style_image = tf.placeholder(tf.float32, shape=img.shape, name='style_image')
		frcs = coarseness(style_image)
		fcon = contrast(style_image)
		fdir = directionality(style_image)
		sess.run(tf.global_variables_initializer())
		print(sess.run(frcs, feed_dict = {style_image:img}))
		print(sess.run(fcon, feed_dict = {style_image:img}))
		print(sess.run(fdir, feed_dict = {style_image:img}))
