###########################################################
# Network in production mode
# FaceEmoji/Precidt
###########################################################

import os
import sys
import json

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

import Lib.utils as utils
import Lib.models as models

class Test:

    def __init__(self):
        """
        Initializer of the tester object
        """

        self.experiment = "model_resnet18_valid-size_0.10_lr_0.00100_b_32_2019-12-01_18-54-50"
        self.model_file = 'model_trained.pwf'
        self.model_file_path = os.path.join(os.getcwd(), "experiments", self.experiment, "models", self.model_file)
        self.face_crop = utils.FaceCrop(reshape=False)

        print(f"Loading checkpoint: {self.model_file}")

        return


    def setup_trained_model(self):
        """
        Method that loads a pretrained model and prepares it for testing
        """

        # setting up device
        self.device = torch.device('cpu')

        # instanciating a new model
        self.model = models.Resnet18(output_dim=10)
        self.model =  self.model.to(self.device)

        print("\n\n")
        print(f"NETWORK STRUCTURE: resnet18\n")
        print(self.model)
        print("\n\n")

        # loading state dictionary
        checkpoint = torch.load(self.model_file_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        return


    def inference(self, img):
        """
        Method that predicts the emojis given a set of images
        """

        label_list = ["angry", "blink", "cow", "happy", "hat", "joon", "monkey", "neutral", "sunglasses", "thinking"]
        img = np.array(Image.open(img))
        labels = []

        with torch.no_grad():

            # getting faces
            faces = self.face_crop.crop_face_from_image(img)
            face_imgs = self.face_crop.get_faces(img, faces)

            for i,face_img in enumerate(face_imgs):

                face_img = face_img[np.newaxis,:,:,:]
                face_img = torch.Tensor(face_img)
                face_img = face_img.transpose(1,3).transpose(2,3)

                output = self.model(face_img)
                output = output.double()

                # saving image with predicitions
                #output = output.numpy()
                predicted_labels = label_list[torch.argmax(output, axis=1)]

                face_img = face_img.numpy()
                face_img = np.transpose(face_img,(0,2,3,1))[0,:,:,0]
                idx = np.argmax(output,axis=1)
                label = label_list[idx]
                labels.append(label)
                # plt.figure()
                # plt.imshow(face_img)
                # plt.title(f"Predicted: {label}")
                #
                # img_path = os.path.join(os.getcwd(),"outputs","inference")
                # dir_existed = utils.create_directory(img_path)
                # plt.savefig( os.path.join(img_path, "img_"+str(utils.timestamp()))+".png" )

            return labels, faces


if __name__ == "__main__":

    images_to_test = [
        "opencv_frame_0.png",
        "opencv_frame_8.png",
        "opencv_frame_20.png",
        "opencv_frame_40.png",
        "opencv_frame_50.png"
    ]

    os.system("clear")

    tester = Test()
    tester.setup_trained_model()

    for image in images_to_test:
        tester.inference(os.path.join("/home/corrales/Femoji/FaceEmoji/Lib/utils/test", image))
