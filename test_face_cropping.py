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


def main():

    # initializing camera
    cam = cv.VideoCapture(0)
    cv.namedWindow("Test Face Cropping")

    # initalizng face cropper
    face_crop = utils.FaceCrop()

    # main loop
    while True:

        # taking image and processing it
        ret, frame = cam.read()
        faces = face_crop.crop_face_from_image(frame)

        # getting faces
        face_crop.get_faces(frame, faces)

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

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


def main2():

    path = os.path.join(os.getcwd(),"Data","sunglasses","1 (13).png")
    img = np.array(Image.open(path))

    plt.figure()
    plt.imshow(img)

    #initalizng face cropper
    face_crop = utils.FaceCrop(reshape=False)

    # getting faces coords
    faces = face_crop.crop_face_from_image(img)

    # getting faces
    face_imgs = face_crop.get_faces(img, faces)

    plt.figure()
    plt.imshow(face_imgs[0])

    plt.show()


if __name__ == "__main__":

    # main()
    main2()


#
