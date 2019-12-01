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

        # training parameters
        self.learning_rate = 0.001
        self.batch_size = 128
        self.max_epochs = 10
        self.valid_size = 0.1
        self.save_frequency = 2

        # relevant paths
        self.root = os.getcwd()
        self.datapath = os.path.join(self.root, "Data")
        self.output_dir = 'model_%s_valid-size_%.2f_lr_%.5f_b_%d' % \
            (model, self.valid_size, self.learning_rate, self.batch_size)
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

        return


    def training_loop(self):
        """
        Computes the training and testing loop
        """

        # training and evaluating model
        for epoch in range(self.max_epochs):
            print("######################################")
            print(f"EPOCH {epoch+1}/{self.max_epochs}")
            print("######################################\n")
            self.test_epoch(epoch)
            self.train_epoch(epoch)
            utils.add_information_to_experiment(self.filepath, self.train_loss, self.valid_loss,
                                                self.train_accuracy, self.valid_accuracy)

        # saving trained model
        trained_model_name = 'model_trained.pwf'
        model_path = os.path.join(self.output_path, 'models')
        dir_existed = utils.create_directory(model_path)
        torch.save( self.model.state_dict(), os.path.join(model_path,trained_model_name))
        print("Training Completed!!")

        return


    def train_epoch(self, epoch):
        """
        Method that computed a training epoch of the network

        Args
        ----
        epoch: Integer
            Current training epoch
        """

        self.model.train()

        #Create list to store loss value to display statistics
        loss_list = []
        accuracy_on_labels = 0
        accuracy_on_true_labels = 0
        label_list = self.dataset.labels

        print('Step Train No: {}'.format(str(epoch+1)))

        # iterate batch by bacth over the train loader
        for i, (images, labels) in enumerate(self.train_loader):

            images = images.to(self.device)
            labels = labels.to(self.device)

            # reseting the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            outputs = outputs.double()

            loss = self.loss_function(input=outputs, target=labels)
            loss_list.append(loss)

            # Backward and optimize
            loss.backward()
            self.optimizer.step()

            # computing accuracy on the training set
            outputs = outputs.cpu().detach().numpy()
            predicted_labels = label_list[np.argmax(outputs, axis=1)]
            accuracy_on_labels += len(np.where(predicted_labels == labels.cpu())[0])

        #Saving the model every 25 epochs
        if(epoch % self.save_frequency == 0):
            trained_model_name = 'model_epoch_' + str(epoch+1) + '.pwf'
            model_path = os.path.join(self.output_path, 'models')
            dir_existed = utils.create_directory(model_path)
            torch.save( self.model.state_dict(), os.path.join(model_path,trained_model_name))

        self.train_accuracy = accuracy_on_labels/self.dataset.train_examples # check for total num of examples
        #Print loss
        self.train_loss = utils.get_loss_stats(loss_list)
        self.train_loss = self.train_loss.item()
        print(f"Train Accuracy: {self.train_accuracy}")

        loss_list=[]
        print("\n")

        return


    def test_epoch(self, epoch):
        """
        Method that computing a validation epoch of the network

        Args
        ----
        epoch: Integer
            Current training epoch
        """

        self.model.eval()

        img_list = []
        accuracy_on_labels = 0
        label_list = self.dataset.labels

        print('Step Valid. No: {}'.format(str(epoch+1)))

        with torch.no_grad():
            loss_list = []
            for i, (images, labels) in enumerate(self.valid_loader):

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                outputs = outputs.double()

                loss_list.append( self.loss_function(input=outputs, target=labels) )

                # computing accuracy on the test set
                outputs = outputs.cpu().detach().numpy()
                predicted_labels = label_list[np.argmax(outputs, axis=1)]
                accuracy_on_labels += len(np.where(predicted_labels == labels.cpu())[0])

                if (i==0 and self.debug):
                    image = images[0:6].cpu().numpy()
                    image = np.transpose(image,(0,2,3,1))
                    output = outputs[0:6]
                    idx = np.argmax(output,axis=1)

                    fig,ax = plt.subplots(2,3)
                    for i in range(6):
                        row = i//3
                        col = i%3
                        ax[row,col].imshow(image[i,:,:,0])
                        ax[row,col].set_title(f"Predicted: {idx[i]}; real: {labels[i]}")

                    img_path = os.path.join(os.getcwd(),"outputs","img")
                    dir_existed = utils.create_directory(img_path)
                    plt.savefig( os.path.join(img_path, "img_epoch_"+str(epoch)))


            self.valid_accuracy = accuracy_on_labels/self.dataset.valid_examples

            self.valid_loss = utils.get_loss_stats(loss_list)
            self.valid_loss = self.valid_loss.item()
            self.loss_over_epochs.append(self.valid_loss)
            print(f"Validation Accuracy: {self.valid_accuracy}")
            print("\n")

            return


if __name__ == "__main__":

    os.system("clear")

    trainer = Trainer()
    trainer.setup_model()
    trainer.training_loop()

#
