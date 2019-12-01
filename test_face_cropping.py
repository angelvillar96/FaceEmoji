###########################################################################
# FaceEmoji/test_face_cropping.py
# This file tests the face cropping functionality using open CV
###########################################################################

import os

import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from PIL import Image

import Lib.utils as utils
from predict import Test as Predict


def main():

    # initializing camera
    cam = cv.VideoCapture(0)
    cv.namedWindow("Test Face Cropping")

    # defining predictor
    predictor = Predict()
    predictor.setup_trained_model()

    # main loop
    while True:

        # taking image and processing it
        ret, frame = cam.read()

        labels, faces = predictor.inference(frame)

        for i in range(len(labels)):
            (x, y, w, h) = faces[i]
            emoji = get_emoji(labels[i], x, y, w, h)
            frame[y:y+h, x:x+w] = emoji

            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv.putText(frame, labels[i], (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv.imshow("Test Face Cropping", frame)
        if not ret:
            break
        k = cv.waitKey(1)

        # when ESC pressed, finish running
        if k%256 == 27:
            print("Escape pressed, closing...")
            break


    # finishing camera
    cam.release()
    cv.destroyAllWindows()


def get_emoji(label, x, y, w, h):

    path = os.path.join(os.getcwd(),"emojies",label+".png")
    size = (h,w)
    img = Image.open(path)
    img = img.resize(size, Image.ANTIALIAS)
    img = np.array(img)[:,:,0:3]

    emoji = np.copy(img)
    emoji[:,:,0] = img[:,:,2]
    emoji[:,:,2] = img[:,:,0]

    return emoji


if __name__ == "__main__":

    main()


#
