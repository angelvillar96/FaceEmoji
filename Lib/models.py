"""
This module loads a pretrained model and manipulate the last layers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torchvision.models as Models


# ====================================================================
#           Classes

class model(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model_layers = self.__load_pretrained_model()



    def __load_pretrained_model(self):
        """
        loads pretrained model
        :return:
        """
        model_full = Models.resnet18(pretrained=True)
        all_layers = list(model_full.children())[:-1]
        all_layers.append(nn.Linear(in_features=512, out_features=10, bias=True))
        model = nn.Sequential(*all_layers)

        return model


    def forward(self, x):
        """

        :param x:
        :return:
        """

        for layer in self.model_layers:
            x = layer(x)


        return x
        
