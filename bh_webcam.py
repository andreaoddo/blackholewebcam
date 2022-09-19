# BlackHoleWebCam
# Author: Andrea Oddo
# Inspired by: Rodriguez C., Marìn C. A. 2017, https://arxiv.org/pdf/1701.04434.pdf

# Needed dependencies:
# - opencv-python
# - numpy
# - scipy

import argparse

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import polyval
from numpy import pi

# Coefficients from [1701.04434]
coefficients = [0.,
                4. / 3.,
                5 * pi / 12. - 4. / 9.,
                122. / 81. - 5 * pi / 18.,
                385. * pi / 576. - 130. / 81.,
                7783. / 2430. - 385 * pi / 432.,
                103565. * pi / 62208. - 21397. / 4374.,
                544045. / 61236. - 85085. * pi / 31104,
                6551545. * pi / 1327104. - 133451. / 8748.,
                1094345069. / 39680928. - 116991875. * pi / 13436928.,
                2268110845. * pi / 143327232 - 1091492587. / 22044960.,
                0.183902,
                0.168300,
                0.155132,
                0.143875,
                0.134145,
                0.125654,
                0.118179,
                0.111548,
                0.105625,
                0.100303]

class BlackHole:
	def __init__(self, position, radius):
		self.x = position[0]
		self.y = position[1]
		self.R = radius

	def impact_parameter(self, x, y):
		return np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

	def critical_impact_parameter(self):
		return 1.5 * np.sqrt(3) * self.R

	def deflection_angle(self, impact_parameter):
		return polyval(1.5 * self.R / impact_parameter, coefficients)

	def compute_distorsion(self, frame_shape):
		thetax = np.fromfunction(lambda i, j: i - self.x, frame_shape, dtype=np.float64)
		thetay = np.fromfunction(lambda i, j: j - self.y, frame_shape, dtype=np.float64)
		distance = np.fromfunction(lambda i, j: np.sqrt((i - self.x) ** 2 + (j - self.y) ** 2),
		                           frame_shape, dtype=np.float64)

		distance[(distance == 0)] = 0.0001

		Idxcrit = (distance <= 1.5 * self.R)
		Omega = self.deflection_angle(distance)
		Omega[Idxcrit] = 0.

		DCx = np.rint(thetax - Omega * thetax * self.R / distance + self.x).astype(np.int16)
		DCy = np.rint(thetay - Omega * thetay * self.R / distance + self.y).astype(np.int16)

		return DCx, DCy, Idxcrit

def parse_arguments():
	parser = argparse.ArgumentParser(description="BlackHole WebCam v1.1")
	parser.add_argument('--cams', help='list available webcams', action="store_true")
	parser.add_argument('-r', help='radius in pixels of the BH', type=float, nargs='?', action='store')
	parser.add_argument('-i', '--id', help='webcam id', type=int, nargs='?', action='store')
	return parser.parse_args()

if __name__ == '__main__':
	import cv2

	bh_args = parse_arguments()

	if bh_args.cams:
		print("Available devices:")
		for cam_idx in range(10):
			cap = cv2.VideoCapture(cam_idx)
			if cap.isOpened():
				print(f"--> Available webcam: {cam_idx}")
				cap.release()
		quit()

	id = bh_args.id if bh_args.id is not None else 1
	cv2.namedWindow("BlackHoleWebCam", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("BlackHoleWebCam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	vc = cv2.VideoCapture(id)
	if vc.isOpened():  # try to get the first frame
		rval, frame = vc.read()
	else:
		rval = False
		quit()

	pixels_x, pixels_y = np.shape(frame)[:2]
	x, y = pixels_x//2, pixels_y//2
	r = bh_args.r if bh_args.r is not None else 20

	black_hole = BlackHole((x, y), r)
	DCx, DCy, Idxcrit = black_hole.compute_distorsion(np.shape(frame)[:2])

	while rval:
		Frame = frame[DCx, DCy]
		Frame[Idxcrit] = [0, 0, 0]
		cv2.imshow("BlackHoleWebCam", Frame)
		rval, frame = vc.read()
		key = cv2.waitKey(20)
		if key == 27:  # exit on ESC
			break

	cv2.destroyWindow("BlackHoleWebCam")
	vc.release()
