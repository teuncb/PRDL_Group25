import tensorflow as tf
from tensorflow import keras
import re
import os
import random
import numpy as np
from lstm import JustLSTM
from sklearn.preprocessing import LabelBinarizer
from eval import evaluate_model
from convLSTM import ConvLSTM
import eval

import data_preprocessing


def extract_label(filename):
    """
    Extract the label out of a filename.

    :param filename: the filename that includes the label.
    :return: the label corresponding to the data in the file.
    """
    filename = filename.split("/")
    pattern = r'_\d'
    # Split the filename based on the regular expression
    split = re.split(pattern, filename[-1])
    # The label is the first item in the filename
    return split[0]


def train_dir(model, model_type, dirpath, epochs, timeframe, label_encoder):
    """
    Trains a model on all files in the given directory.

    :param model: the model that needs to be trained.
    :param model_type: whether the model is an 'lstm' or a 'convlstm'.
    :param dirpath: the path to the training data.
    :param epochs: how many epochs we would like to fit the model on this directory.
    :param timeframe: how many timesteps are included in one window.
    :param label_encoder: used to transform the textual labels into numeric ones.
    :return: Void
    """
    # Get a list of all filenames
    dirnames = os.listdir(dirpath)

    random.shuffle(dirnames)
    # Int divide by eight to avoid float errors
    batches = len(dirnames) // 8

    # Go over all files in the directory for the amount of epochs that was specified
    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch + 1))
        print()

        # Fit on 8 files at a time (one file is considered one batch)
        for i in range(batches):
            fit_list = []
            print("Fitting on files {}-{}....".format((i * 8), ((i + 1) * 8)))

            files = dirnames[i * 8:(i + 1) * 8]
            # Load all the data from the 8 files and extract their corresponding labels
            for j in range(8):
                filename = dirpath + "/" + files[j]

                data = data_preprocessing.read_prepro_file(filename)
                label = extract_label(filename)

                fit_list.append((data, label))


            x_temp = []
            y_temp = []

            # Preprocess the data from the current 8 files
            for counter, (data, label) in enumerate(fit_list):
                # Creates windows of length 'timeframe' to fit the model on
                windows = data_preprocessing.create_windows(data, model_type, timeframe)

                # Will temporarily store the windows
                x_windows = []

                for window in windows:
                    if model_type == "convlstm":
                        # The windows are in shape (20, 21, timeframe), but the ConvLSTM network needs the timeframe first in the shape
                        res = np.reshape(window, (timeframe, 20, 21))
                        x_windows.append(np.expand_dims(res, axis=-1))
                    else:
                        x_windows.append(np.transpose(window))

                # Store all the windows of the current file in x_temp
                x_temp.extend(x_windows)

                # Encode the textual label to one-hot-encoding
                encoded_label = label_encoder.transform([label])

                y_current = [np.squeeze(encoded_label)] * len(x_windows)
                y_temp.extend(y_current)

            # Stack all windows of the 8 files to gain a training batch
            x_train = np.stack(x_temp, axis=0)
            y_train = np.stack(y_temp, axis=0)

            # Fit the model on the data of the current batch (8 files)
            model.fit(x_train, y_train, epochs=1)

            print("--------------------------------------------------------------")

    # Save the model after training
    model.save("trained_LSTM_tf10_e4_50lstm_100ep_intra")
