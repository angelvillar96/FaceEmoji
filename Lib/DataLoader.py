"""
This class loades the data!
"""
import os

import numpy as np
from PIL import Image
from scipy import misc
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import Lib.utils as utils



# ========================================================================
#               CLASSES

class Dataset(Dataset):

    def __init__(self, data_dir='', use_gpu=True, train_size=0.8, valid_size=0.2
                 , shuffle=True, batch_size=128, seed=13, debug=False):

        super().__init__()
        self.random_seed = seed
        self.use_gpu = use_gpu
        self.shuffle = shuffle
        self.debug = debug

        self.data_dir = data_dir
        self.train_size = train_size
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.file_paths = []
        self.labels = []

        self.face_crop = utils.FaceCrop(reshape=False)

        # reading the directories and labels!
        self.__load_dirs()


    # =====================================================================
    def __load_dirs(self):
        """
        loads all labels and directories!
        :return:
        """
        all_labels_ = os.listdir(self.data_dir)
        labels = [label for label in all_labels_ if not '.' in label]

        file_lists = [os.listdir(os.path.join(self.data_dir, label)) for label in labels]

        label_strings = []
        files = []
        for i, file_list in enumerate(file_lists):
            label_strings += len(file_list) * [labels[i]]
            files += file_list

        self.file_paths = [os.path.join(self.data_dir, label_strings[i], file) for i,file in enumerate(files)]


        # changing the labels to 0 and 1!
        labels_unique = ['angry', 'blink', 'cow', 'happy', 'hat',
                         'joon', 'monkey', 'neutral', 'sunglasses',
                         'thinking']

        self.labels = [labels_unique.index(ll) for ll in label_strings]



    def __len__(self):
        return self.shape


    def __getitem__(self, idx):
        images, labels = self.__get_images_and_labels(idx)

        return images, labels


    def get_images_given_paths(self, paths, labels):
        """
        """

        images = np.empty((0,224,224,3))
        new_labels = []

        # processing images in the batch
        for i, path in enumerate(paths):

            # reading image and getting face coords
            image = np.array(Image.open(path))
            faces = self.face_crop.crop_face_from_image(image)

            # getting faces
            face_imgs = self.face_crop.get_faces(image, faces)

            # discarding if no face was found
            if(len(face_imgs)==1):
                try:
                    images = np.concatenate((images, face_imgs[0][np.newaxis,:,:,:]), axis=0)
                    new_labels.append(labels[i])
                except:
                    pass

        while(images.shape[0]<self.batch_size):
            images = np.concatenate((images, images[-1][np.newaxis,:,:,:]), axis=0)
            new_labels.append(new_labels[-1])

        #torch.tensor
        images = torch.Tensor(images)
        images = images.transpose(1,3).transpose(2,3)
        new_labels = torch.Tensor(new_labels)

        return images, new_labels


    # ======================================================
    # ======================================================
    def __get_images_and_labels(self, idx):
        """
        gets the images and labels with the specified indexes
        :return:
        """
        # if not type(idx) == 'list':
            # idx = [idx]

        # print(idx)
        # images = np.asarray([misc.imread(self.file_paths[i]) for i in idx])
        # images = np.asarray(misc.imread(self.file_paths[idx]))
        images = np.array(Image.open(self.file_paths[idx]))
        labels = self.labels[idx]

        return images, labels


    # ========================================================================
    # ========================================================================
    def get_train_validation_set(self):
        """
        Creates the train/validation split and returns the data loaders
        """

        num_train = len(self.file_paths)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        # randomizing train and validation set
        if(self.shuffle):
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        # getting idx for train and validation
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # creating data loaders
        self.pairs = list(zip(self.file_paths, self.labels))
        train_loader = torch.utils.data.DataLoader(self.pairs,
                    batch_size=self.batch_size, sampler=train_sampler)

        valid_loader = torch.utils.data.DataLoader(self.pairs,
                        batch_size=self.batch_size, sampler=valid_sampler)

        self.train_examples = len(train_idx)
        self.valid_examples = len(valid_idx)

        if(self.debug):
            print("\n")
            print(f"Total number of examples is {num_train}")
            print(f"Size of training set is approx {self.train_examples}")
            print(f"Size of validation set is approx {self.valid_examples}")

        return train_loader, valid_loader



    # ========================================================================
    # ========================================================================
    def get_test_set(self):
        pass


if __name__ == '__main__':

    datadir = 'D:\MS II\III\emojies\FaceEmoji\Data\Data'
    ds = Dataset(data_dir=datadir)
    ds.get_train_validation_set()
