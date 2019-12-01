"""
This class loades the data!
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from scipy import misc




# ========================================================================
#               CLASSES

class Dataset(Dataset):

    def __init__(self, data_dir='', use_gpu=True, train_size=0.8, valid_size=0.2
                 , shuffle=True, batch_size=128, seed=13, debug=False):

        super().__init__()
        self.data_dir = data_dir
        self.use_gpu = use_gpu
        self.train_size = train_size
        self.valid_size = valid_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.random_seed = seed
        self.dirs = None
        self.labels = None
        self.alllabels = None
        self.debug = debug
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
        dirs = [os.listdir(os.path.join(self.data_dir, label)) for label in labels]

        self.labels = []
        self.dirs = []
        for i, drs in enumerate(dirs):
            self.labels += len(drs) * [labels[i]]
            self.dirs += drs

        # changing the labels to 0 and 1!
        labels_unique = ['angry', 'blink', 'cow', 'happy', 'hat',
                         'joon', 'monkey', 'neutral', 'sunglasses',
                         'thinking']

        d = [labels_unique.index(ll) for ll in self.labels]
        self.labels = d


    def __len__(self):
        return self.shape


    def __getitem__(self, idx):
        images, labels = self.__get_images_and_labels(idx)

        return images, labels



    # ======================================================
    # ======================================================
    def __get_images_and_labels(self, idx):
        """
        gets the images and labels with the specified indexes
        :return:
        """
        if not type(idx) == 'list':
            idx = list(idx)

        images = np.asarray([misc.imread(self.dirs[i]) for i in idx])
        labels = self.labels[idx]

        return images, labels


    # ========================================================================
    # ========================================================================
    def get_train_validation_set(self):
        """
        Creates the train/validation split and returns the data loaders
        """

        num_train = len(self.dirs)
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
        train_loader = torch.utils.data.DataLoader(self.dirs,
                    batch_size=self.batch_size, sampler=train_sampler)

        valid_loader = torch.utils.data.DataLoader(self.dirs,
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

