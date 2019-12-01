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

class Resnet18(nn.Module):

    def __init__(self, output_dim=10):
        super(Resnet18, self).__init__()

        self.output_dim = output_dim
        self.model_layers = self.__load_pretrained_model()
        return


    def __load_pretrained_model(self):
        """
        loads pretrained model
        :return:
        """
        model_full = Models.resnet18(pretrained=True)
        all_layers = list(model_full.children())[:-1]
        all_layers.append(nn.Linear(in_features=512, out_features=self.output_dim, bias=True))
        model = nn.Sequential(*all_layers)

        return model


    def forward(self, x):
        """

        :param x:
        :return:
        """

        for layer in self.model_layers[:-1]:
            x = layer(x)
        x = x.view(-1, 512)
        x = self.model_layers[-1](x)

        return x


#
