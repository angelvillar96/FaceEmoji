# Face Emoji
### By Angel Villar-Corrales and Mohammad J.

This project uses image processing techniques to perform face detection and deep learning in order to replace the faces by an emoji similar to the detected facial expression.

## Getting Started

To get this repository, fork this repository or clone it using the following command:

>git clone https://github.com/angelvillar96/FaceEmoji.git

### Prerequisites

To get the repository running, you will need the following packages: numpy, matplotlib, openCV and pyTorch

You can obtain them easily by installing the conda environment file included in the repository. To do so, run the following command from the Conda Command Window:

```shell
$ conda env create -f environment.yml
$ activate FaceEmoji
```

*__Note__:* This step might take a few minutes


## Contents

### Face Detection

In order to perform face detection, a cascade classifier from openCV has been used. This classifier model (based on Haar transform) was pretrained on frontal face images, therefore not needing further training. More information can be found in the following website: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

### Deep Learning Model

The deep learning part of the project has been developed using the PyTorch Library. The *emotion detection* has been preformed using a Resnet18 model (see https://arxiv.org/pdf/1512.03385.pdf for further information about Residual networks). The model used was pretrained on the ImageNet dataset containing over 1000000 images.

In order to adapt this model to our particular task, traser learning has been applied.

On the one hand, the convolutional part of the network, which performs feature extraction, has been kept.

On the other hand,  the fully-connected part, which performs classification, has been modified. The last fully-connected layer was replaced for a new layer tailormade for our purpose.

### Dataset
