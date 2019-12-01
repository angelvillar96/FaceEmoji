###########################################################
# Training the Network to detect emotions
# FaceEmoji/Train
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

class Trainer:

    def __init__(self):
        """
        Initializing trainer
        """

        self.root = os.getcwd()
        self.datapath = os.path.join(self.root, "Data")

        # training parameters
        self.learning_rate = 0.001
        self.batch_size = 128
        self.max_epochs = 10
        self.valid_size = 0.1

        # defining output path and output folder
        self.output_dir = 'model_%s_valid-size_%.2f_lr_%.5f_b_%d' % \
            (model, self.valid_size, self.learning_rate, self..batch_size)
        self.output_path = os.path.join(os.getcwd(), "experiments", self.output_dir + '_' +
                                   utils.timestamp())
        dir_existed = utils.create_directory(output_path)

        # creating the experiment file with metadata and so on
        self.filepath = utils.create_experiment(self.output_path, model="resnet", valid_size=self.valid_size,
                                                learning_rate=self.learning_rate, batch_size=self.batch_size,
                                                max_epochs=self.max_epochs)



    def setup_model(self):
        """
        Sets up network, dataloader, optimizers and so on
        """

        # fetching the dataset
        self.dataset = Dataset(data_path=self.data_path, use_gpu=True, dataset=self.dataset,
                               train=True, debug=self.debug, valid_size=self.valid_size,
                               batch_size=self.batch_size, shuffle=True)

        self.train_loader, self.valid_loader = self.dataset.get_train_validation_set()

        # setting up device
        torch.backends.cudnn.fastest = True
        torch.cuda.device(cuda_device_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ToDo
        # here we load the model

        model_architecture = str(model)
        print("\n\n")
        print(f"NETWORK STRUCTURE: {self.model_type}\n")
        print(model)
        print("\n\n")

         # setting up model parameters
        self.optimizer= torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_function = torch.nn.CrossEntropyLoss()

         # saving the network architecture in the metadata file
        utils.save_network_architecture(self.filepath, model_architecture, self.optimizer,
                                        self.loss_function)





def __name__ == "__main__":

    os.system("clear")

    trainer = Trainer()
    trainer.setup_model()
    trainer.training_loop()

#
