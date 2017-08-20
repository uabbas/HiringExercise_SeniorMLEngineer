from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
import argparse
import cv2
import numpy as np
import os
import random
import sys
import threading

# 0 - vgg16
# 1 - vgg19
# 2 - resnet50
# 3 - inceptionv3
# 4 - mobilenet

def createmodel(type):
    if type == 0:
        model = VGG16(weights="imagenet")
    elif type == 1:
        model = VGG19(weights="imagenet")
    elif type == 2:
        model = ResNet50(weights="imagenet")
    elif type == 3:
        model = InceptionV3(weights="imagenet")
    else:
        model = MobileNet(weights="imagenet")
    return model


frame = None
is_webcam = False
drop_rate = 10
modeltype = 3
model = createmodel(modeltype)
cap = None

if (is_webcam):
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("D:/downloads/nature2.mp4")

if cap.isOpened():
    print("Camera OK")
else:
    cap.open()

while (True):
    for i in range(0, drop_rate):
        ret, original = cap.read()
        if (original is None):
            if (not is_webcam):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, original = cap.read()

    if (modeltype == 3):
        framest = cv2.resize(original, (299, 299))
    else:
        framest = cv2.resize(original, (224, 224))
    image = cv2.cvtColor(framest, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = image.reshape((1,) + image.shape)
    image = preprocess_input(image, 'channels_last')
    preds = model.predict(image)
    decoded_preds = decode_predictions(preds)
    first_pred = decoded_preds[0]
    pred = first_pred[0]
    label = pred[1]

    # Display the predictions
    # print("ImageNet ID: {}, Label: {}".format(inID, label))
    cv2.putText(original, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", original)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cv2.destroyAllWindows()
cap.release()
frame = None
sys.exit()