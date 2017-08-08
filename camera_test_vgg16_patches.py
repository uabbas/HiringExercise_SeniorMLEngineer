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

label = ['', 0, 0, 200, 200]
frame = None
result_box = None

WINDOW_SIZES = [i for i in range(400, 800, 400)]

class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global result_box
        # Load the VGG16 network
        print("[INFO] loading network...")
        self.model = VGG16(weights="imagenet")

        while (~(frame is None)):
            self.get_best_bounding_box(frame)

    def get_best_bounding_box(self, img, step=50, window_sizes=WINDOW_SIZES):
        global label
        global result_box
        best_box = None
        best_box_prob = -np.inf
        best_result = None

        # loop window sizes: 20x20, 30x30, 40x40...160x160
        for win_size in window_sizes:
            for top in range(0, img.shape[0] - win_size + 1, step):
                for left in range(0, img.shape[1] - win_size + 1, step):
                    # compute the (top, left, bottom, right) of the bounding box
                    box = (top, left, top + win_size, left + win_size)

                    # crop the original image
                    cropped_img = img[box[0]:box[2], box[1]:box[3]]

                    # predict how likely this cropped image is dog and if higher
                    # than best save it
                    #print('predicting for box %r' % (box,))
                    resized_cropped_img = cv2.resize(cropped_img, (224, 224))
                    result = self.predict(resized_cropped_img)
                    box_prob = result[2]
                    if box_prob > best_box_prob:
                        best_box = box
                        best_box_prob = box_prob
                        best_result = result

        label[0] = best_result[1]
        label[1] = best_box[0]
        label[2] = best_box[1]
        label[3] = best_box[2]
        label[4] = best_box[3]
        result_box = best_box

    def predict(self, cropped_img):
        global label
        image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image.reshape((1,) + image.shape)
        image = preprocess_input(image, 'channels_last')
        preds = self.model.predict(image)
        decoded_preds = decode_predictions(preds)
        first_pred = decoded_preds[0]
        pred = first_pred[0]
        return pred


cap = cv2.VideoCapture(0)
if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

keras_thread = MyThread()
keras_thread.start()

while (True):
    ret, original = cap.read()

    frame = original #cv2.resize(original, (224, 224))
    #	t = MyThread()
    #	t.run()
    # Display the predictions
    # print("ImageNet ID: {}, Label: {}".format(inID, label))
    draw_box = (label[1], label[2], label[3], label[4])
    draw_label = label[0]
    cv2.putText(original, "Label: {}".format(draw_label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    #if (draw_box is not None):
    cv2.rectangle(original, (draw_box[0], draw_box[1]), (draw_box[2], draw_box[3]), (0, 0, 255), 5);
    cv2.imshow("Classification", original)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()