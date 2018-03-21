import cv2
import numpy as np

def coarseness(image, kmax):
	image = np.array(image)
	w = image.shape[0]
	h = image.shape[1]
	kmax = kmax if (np.power(2,kmax) < w) else int(np.log(w) / np.log(2))
	kmax = kmax if (np.power(2,kmax) < h) else int(np.log(h) / np.log(2))
	average_gray = np.zeros([w,h,kmax])
	horizon = np.zeros([kmax,w,h])
	vertical = np.zeros([kmax,w,h])
	Sbest = np.zeros([w,h])

	for k in range(kmax):
		window = np.power(2,k)
		for wi in range(w)[window:(w-window)]:
			for hi in range(h)[window:(h-window)]:
				average_gray[k][wi][hi] = np.sum(image[wi-window:wi+window, hi-window:hi+window])
		for wi in range(w)[window:(w-window-1)]:
			for hi in range(h)[window:(h-window-1)]:
				horizon[k][wi][hi] = average_gray[k][wi+window][hi] - average_gray[k][wi-window][hi]
				vertical[k][wi][hi] = average_gray[k][wi][hi+window] - average_gray[k][wi][hi-window]
		horizon[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))
		vertical[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))

	for wi in range(w):
		for hi in range(h):
			h_max = np.max(horizon[:,wi,hi])
			h_max_index = np.argmax(horizon[:,wi,hi])
			v_max = np.max(vertical[:,wi,hi])
			v_max_index = np.argmax(vertical[:,wi,hi])
			index = h_max_index if (h_max > v_max) else v_max_index
			Sbest[wi][hi] = np.power(2,index)

	fcrs = np.mean(Sbest)
	return frcs


def contrast(image):
	pass

def directionality(image):
	pass

def linelikeness(image, sita, dist):
	pass

def regularity(image, filter):
	pass

def roughness(fcrs, fcon):
	return fcrs + fcon

if __name__ == '__main__':
	'''
	img = cv2.imread('rain_princess.jpg',cv2.IMREAD_GRAYSCALE)
	fcrs = coarseness(img, 5)
	fcon = contrast(img)
	fdir, sita = directionality(img)
	flin = linelikeness(img,sita,4)
	freg = regularity(img,64)
	frgh = roughness(fcrs, fcon)
	printf("coarseness: %d" % fcrs);
	printf("contrast: %d" % fcon)
	printf("directionality: %d" % fdir)
	printf("linelikeness: %d" % flin)
	printf("regularity: %d" % freg)
	printf("roughness: %d" % frgh)
	'''