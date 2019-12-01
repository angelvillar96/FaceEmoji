###########################################################
# Network in production mode
# FaceEmoji/Precidt
###########################################################

import os
import sys
import json

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

import Lib.utils as utils


class Test:

    def __init__(self):
        """
        Initializer of the tester object
        """

        self.experiment = ""
        self.model = 'model_trained.pwf'
        self.model_file_path = os.path.join(os.getcwd(), "experiments", self.experiment, "models", self.mode)

        if(self.debug):
            print(f"Loading checkpoint: {self.model_file}")

        return


    def setup_trained_model(self):
        """
        Method that loads a pretrained model and prepares it for testing
        """

        # ToDo
        # instanciating a new model


        print("\n\n")
        print(f"NETWORK STRUCTURE: {self.model_type}\n")
        print(model)
        print("\n\n")

        # loading state dictionary
        checkpoint = torch.load(self.model_file_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        return


    def inference(self, images):
        """
        Method that predicts the emojis given a set of images
        """

        self.model.eval()
        label_list = ["angry", "blink", "cow", "happy", "hat", "joon", "monkey", "neutral", "sunglasses", "thinking"]

        with torch.no_grad():

            for img in images:

                outputs = self.model(img)
                outputs = outputs.double()

                # saving image with predicitions
                outputs = outputs.detach().numpy()
                predicted_labels = label_list[np.argmax(outputs, axis=1)]

                img = img.numpy()
                image = np.transpose(image,(0,2,3,1))
                output = outputs[0:6]
                idx = np.argmax(output,axis=1)
                label = label_list[idx]

                plt.figure()
                plt.imshow(img)
                plt.title(f"Predicted: {label}")

                img_path = os.path.join(os.getcwd(),"outputs","img")
                dir_existed = utils.create_directory(img_path)
                plt.savefig( os.path.join(img_path, "img_epoch_"+str(epoch)))

            return


if __name__ == "__main__":

    images_to_test = [
        "",
        ""
    ]

    os.system("clear")

    tester = Test()
    tester.setup_trained_model()
    tester.inference(images_to_test)
