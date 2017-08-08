from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.inception_v3 import InceptionV3
import argparse
import cv2
import numpy as np
import os
import random
import sys

import threading

label = ['']
frame = None

def predictst(framest):
	global label
	model = InceptionV3(weights="imagenet")
	image = cv2.cvtColor(framest, cv2.COLOR_BGR2RGB).astype(np.float32)
	image = image.reshape((1,) + image.shape)
	image = preprocess_input(image, 'channels_last')
	preds = model.predict(image)
	decoded_preds = decode_predictions(preds)
	first_pred = decoded_preds[0]
	pred = first_pred[0]
	label[0] = pred[1]
	return label


class MyThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		#global label
		# Load the VGG16 network
		print("[INFO] loading network...")
		self.model = InceptionV3(weights="imagenet")
		while (~(frame is None)):
			self.predict(frame)

	def predict(self, frame):
		global label
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
		image = image.reshape((1,) + image.shape)
		image = preprocess_input(image, 'channels_last')
		preds = self.model.predict(image)
		decoded_preds = decode_predictions(preds)
		first_pred = decoded_preds[0]
		pred = first_pred[0]
		label[0] = pred[1]
		return label

cap = cv2.VideoCapture(0)
if (cap.isOpened()):
	print("Camera OK")
else:
	cap.open()

keras_thread = MyThread()
keras_thread.start()

while (True):
	ret, original = cap.read()

	frame = cv2.resize(original, (299, 299))
	# Display the predictions
	# print("ImageNet ID: {}, Label: {}".format(inID, label))
	cv2.putText(original, "Label: {}".format(label[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
	cv2.imshow("Classification", original)

	if (cv2.waitKey(1) & 0xFF == ord('q')):
		break;

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()