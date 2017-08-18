from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
import argparse
import cv2
import numpy as np
import os
import random
import sys

import threading

#import tensorflow as tf
#sess = tf.Session()

#from keras import backend as K
#K.set_session(sess)

label = ['']
frame = None

class MyThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		#global label
		# Load the VGG16 network
		print("[INFO] loading network...")
		self.model = VGG16(weights="imagenet")

		while (~(frame is None)):
			result = self.predict(frame)
			#if (len(result) >= 1):
			#	label = result[1]

	def predict(self, frame):
		global label
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
		#image = image.transpose((2, 0, 1))
		image = image.reshape((1,) + image.shape)
		
		#image = (np.array(frame).astype(np.float32)).reshape((1, 3, 224, 224))
		
		#image = np.expand_dims(image, axis=0)
		#image=np.swapaxes(np.swapaxes(frame, 1, 2), 0, 1).astype(np.float32)
		#image = np.expand_dims(image, axis=0)

		image = preprocess_input(image, 'channels_last')
		preds = self.model.predict(image)
		decoded_preds = decode_predictions(preds)
		first_pred = decoded_preds[0]
		pred = first_pred[0]
		label[0] = pred[1]
		return label

is_webcam = True

cap = None
if (is_webcam):
	cap = cv2.VideoCapture(0)
else:
	cap = cv2.VideoCapture("D:/downloads/earth.mp4")

if (cap.isOpened()):
	print("Camera OK")
else:
	cap.open()

keras_thread = MyThread()
keras_thread.start()

while (True):
	ret, original = cap.read()
	if (original is None):
		if (not is_webcam):
			cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			ret, original = cap.read()

	frame = cv2.resize(original, (224, 224))
#	t = MyThread()
#	t.run()
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