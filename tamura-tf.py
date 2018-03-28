import tensorflow as tf
import numpy as np

def coarseness(image, kmax):
	h = image.shape[0]
	w = image.shape[1]
	c = image.shape[2]
	kmax = kmax if (np.power(2,kmax) < w) else int(np.log(w) / np.log(2))
	kmax = kmax if (np.power(2,kmax) < h) else int(np.log(h) / np.log(2))

	'''
	average_gray = np.zeros([h,w,c,kmax])
	horizon = np.zeros([kmax,h,w])
	vertical = np.zeros([kmax,h,w])
	Sbest = np.zeros([h,w])

	for k in range(kmax):
		window = np.power(2,k)
		for hi in range(h)[window:(h-window)]:
			for wi in range(w)[window:(w-window)]:
				average_gray[k][hi][wi] = np.sum(image[hi-window:hi+window, wi-window:wi+window])
		for hi in range(h)[window:(h-window-1)]:
			for wi in range(w)[window:(w-window-1)]:
				horizon[k][hi][wi] = average_gray[k][hi+window][wi] - average_gray[k][hi-window][wi]
				vertical[k][hi][wi] = average_gray[k][hi][wi+window] - average_gray[k][hi][wi-window]
		horizon[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))
		vertical[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))

	for hi in range(h):
		for wi in range(w):
			h_max = np.max(horizon[:,hi,wi])
			h_max_index = np.argmax(horizon[:,hi,wi])
			v_max = np.max(vertical[:,hi,wi])
			v_max_index = np.argmax(vertical[:,hi,wi])
			index = h_max_index if (h_max > v_max) else v_max_index
			Sbest[hi][wi] = np.power(2,index)

	fcrs = np.mean(Sbest)
	return frcs
	'''
	'''
	average_gray = np.zeros([h,w,c,kmax])
	horizon = np.zeros([kmax,h,w,c])
	vertical = np.zeros([kmax,h,w,c])
	Sbest = np.zeros([h,w,c])
	for ci in range(c):
		for ki in range(kmax):
			window = np.power(2,k)
			weight_window = tf.constant(np.ones([window, window]))
			average_gray[:,:,ci,ki] = tf.nn.conv2d(image[:,:,ci], weight_window, [1,1,1,1], padding='SAME')
			weight_horizon = tf.constant()
			weight_vertical = tf.constant()
	'''
	average_gray = np.zeros([kmax,h,w,c])
	horizon = np.zeros([kmax,h,w,c])
	vertical = np.zeros([kmax,h,w,c])
	Sbest = np.zeros([h,w,c])

	for ci in range(c):
		for k in range(kmax):
			window = tf.pow(2,k)
			for hi in range(h)[window:(h-window)]:
				for wi in range(w)[window:(w-window)]:
					average_gray[k][hi][wi][ci] = tf.reduce_sum(image[hi-window:hi+window, wi-window:wi+window,ci])
			for hi in range(h)[window:(h-window-1)]:
				for wi in range(w)[window:(w-window-1)]:
					horizon[k][hi][wi][ci] = tf.subtract(average_gray[k][hi+window][wi][ci], average_gray[k][hi-window][wi][ci])
					vertical[k][hi][wi][ci] = tf.subtract(average_gray[k][hi][wi+window][ci], average_gray[k][hi][wi-window][ci])
			horizon[k] = tf.div(horizon[k], tf.pow(2, 2*(k+1)))
			vertical[k] = tf.div(horizon[k], tf.pow(2, 2*(k+1)))

	for hi in range(h):
		for wi in range(w):
			h_max = np.max(horizon[:,hi,wi])
			h_max_index = np.argmax(horizon[:,hi,wi])
			v_max = np.max(vertical[:,hi,wi])
			v_max_index = np.argmax(vertical[:,hi,wi])
			index = h_max_index if (h_max > v_max) else v_max_index
			Sbest[hi][wi] = tf.pow(2,index)

	fcrs = np.mean(Sbest)
	return frcs