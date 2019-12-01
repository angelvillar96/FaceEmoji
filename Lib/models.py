"""
This module loads a pretrained model and manipulate the last layers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy



# ====================================================================
#           Classes

class model(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = self.__load_pretrained_model()



    def __load_pretrained_model(self):
        pass

    def forward(self):
        pass
