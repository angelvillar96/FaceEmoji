###########################################################################
# FaceEmoji/Lib/utils/face_cropping.py
# Methods use to crop faces from an image
###########################################################################

import os

import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

class FaceCrop:

    def __init__(self, reshape=True, size=224, offset=50):
        """
        Object initializer

        Args:
        ----
        reshape: Boolean
            Whether the cropped faces should be reshaped
        size: Tuple (H,W)
            Size to reshape the faces to
        offset: Integer
            offset in percentage to extend from the face
        """

        # loading cascade detector
        path_to_xml = os.path.join(os.getcwd(), "Lib", "utils", 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv.CascadeClassifier(path_to_xml)
        self.reshape = reshape
        self.size = size
        self.offset = offset


    def crop_face_from_image(self, img):
        """
        Method that crops the faces from an image

        Args:
        -----
        img: Numpy Array (C,X,Y)
            Image to be cropped
        """

        # converting image to gray and enhancing
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_gray = cv.equalizeHist(img_gray)

        # detecting faces
        faces = self.face_cascade.detectMultiScale(img_gray,
                                                   scaleFactor=1.3,
                                                   minNeighbors=2,
                                                   minSize=(20, 20))
        return faces


    def get_faces(self, img, faces):
        """
        Method that creates new images with just the faces
        """

        face_imgs = []

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:

            offset = int(np.floor((self.size-w)/2))
            left = max(x-offset, 0)
            top = max(y-offset, 0)

            if( offset*2+w != 224 ):
                offset+=1

            right = min(x+w+offset, img.shape[1])
            bottom = min(y+h+offset, img.shape[0])

            face_img = img[top:bottom, left:right, :]
            new_face_img = np.copy(face_img)

            if(self.reshape):
                new_face_img[:,:,0] = face_img[:,:,2]
                new_face_img[:,:,2] = face_img[:,:,0]

            face_imgs.append(new_face_img)

        return face_imgs

#
