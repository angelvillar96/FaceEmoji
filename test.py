#
#
#
import os

import Lib.DataLoader as Datasets

os.system("clear")
data_dir = os.path.join(os.getcwd(), "Data")

dataset = Datasets.Dataset(data_dir=data_dir, batch_size=2)
train_loader, valid_loader = dataset.get_train_validation_set()

for i, (x,y) in enumerate(valid_loader):
    print(f"Batch {i}")

    print(f"Paths: {x}")
    print(f"Labels {y}")

    images, labels = dataset.get_images_given_paths(x, y)
    break


# print(dataset[0])
