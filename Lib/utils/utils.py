###########################################################
# Useful methods for several purposes
# FaceEmoji/Lib/Utils/Utils
###########################################################
import os
import json
import datetime

import torch


def timestamp():
    """
    Computes and returns current timestamp

    Args:
    -----
    timestamp: String
        Current timestamp in formate: hh-mm-ss
    """

    timestamp = str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')
    return timestamp


def create_directory(path):
    """
    Method that creates a directory if it does not already exist

    Args
    ----
    path: String
        path to the directory to be created
    dir_existed: boolean
        Flag that captures of the directory existed or was created
    """
    dir_existed = True
    if not os.path.exists(path):
        os.makedirs(path)
        dir_existed = False
    return dir_existed


def get_loss_stats(loss_list):
    """
    Computes loss statistics given a list of loss values

    TODO: SAVE LOSS INTO COMPRESSED NPZ OR WRITE INTO FILE
    FOR FURTHER PROCESSING

    Args:
    -----
    loss_list: List
        List containing several loss values
    """

    if(len(loss_list)==0):
        return

    loss_np = torch.stack(loss_list)
    avg_loss = torch.mean(loss_np)
    max_loss = torch.max(loss_np)
    min_loss = torch.min(loss_np)
    print(f"Average loss: {avg_loss} -- Min Loss: {min_loss} -- Max Loss: {max_loss}")

    return avg_loss


def create_experiment(output_path, output_dir, model, valid_size, learning_rate,
                      batch_size, max_epochs, **kwargs):
    """
    Creates a json file with metadata about the experiment

    Args:
    -----
    output_name: String
        Path to the experiment folder
    output_dir: String
        Folder where the outputs experiments will be saved
    model_type: String
        name of the model architecture
    valid_size: float [0-1]
        amount of corruption added to the training labels
    learning_rate: Float
        learning rate
    batch_size: Int
        batch size
    max_epochs: Int
        Maximum nuber of epochs to be executed
    """

    filepath = os.path.join(output_path, "experiment_data.json")

    data = {}
    data["experiment_started"] = timestamp()
    data["learning_rate"] = learning_rate
    data["batch_size"] = batch_size
    data["valid_size"] = valid_size
    data["max_epochs"] = max_epochs

    # model information
    data["model"] = {}
    data["model"]["model_name"] = model

    # loss and accuracy information
    data["loss"] = {}
    data["loss"]["train_loss"] = []
    data["loss"]["valid_loss"] = []
    data["accuracy"] = {}
    data["accuracy"]["train_accuracy"] = []
    data["accuracy"]["valid_accuracy"] = []


    with open(filepath, "w") as file:
        json.dump(data, file)

    return filepath


def save_network_architecture(filepath, architecture, loss_function, optimizer):
    """
    Saves the network architecture (layers, params, optimizer, ...) into the metadata file
    Creates an extra file to dump the architecture in a human-readable way
    """

    # saving architecture, loss and optimizer in the metadata file
    with open(filepath) as filedata:
        data = json.load(filedata)

    data["model"]["model_architecture"] = architecture
    data["optimizer"] = optimizer.__class__.__name__
    data["loss_function"] = loss_function.__class__.__name__

    with open(filepath, "w") as file:
        json.dump(data, file)

    # creating file and saving architecture in a human-readable way
    directory = "/".join(filepath.split("/")[:-1])
    new_file = os.path.join(directory, "network_architecture.txt")

    f = open(new_file, "w")
    f.write(architecture)
    f.write("\n\n\n")
    f.close()

    return


def add_information_to_experiment(filepath, train_loss=None, valid_loss=None, train_accuracy=None,
                                  valid_accuracy=None):
    """
    Adds information to the JSON data
    """

    with open(filepath) as filedata:
        data = json.load(filedata)

    data["loss"]["train_loss"].append(train_loss)
    data["loss"]["valid_loss"].append(valid_loss)
    data["accuracy"]["train_accuracy"].append(train_accuracy)
    data["accuracy"]["valid_accuracy"].append(valid_accuracy)

    with open(filepath, "w") as file:
        json.dump(data, file)

    return


#
