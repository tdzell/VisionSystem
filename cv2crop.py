import numpy as np
import cv2
import glob

for f in glob.glob('*.jpg'):
	img = cv2.imread(f, 1)
	img = img[0:544, 150:694]
	cv2.imwrite(f, img)