import load_data
import downsampling
import re
import random
import os
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from cascadeNet import CascadeNet
from create_windows import create_windows
from sklearn.preprocessing import LabelBinarizer
from training import train_batch

i_train_path = "Final Project data/Intra/train"
i_train_prepro_path = "Final Project data/Intra/train_prepro"
i_test_path = "Final Project data/Intra/test"

c_train_path = "Final Project data/Cross/train"
c_train_prepro_path = "Final Project data/Cross/train_prepro"
c_test1_path = "Final Project data/Cross/test1"
c_test2_path = "Final Project data/Cross/test2"
c_test3_path = "Final Project data/Cross/test3"


epochs = 1
timeframe = 20 # amount of timesteps included in one 'window' --> is 5 in paper by Zhang et al.
output_classes = 4

cnn_input_shape = (20, 21, 1)
rnn_input_shape = (timeframe, 64)    # was 1024 in originele paper

# TODO: kijken of we de input als tensors willen doen (wat wel netter is I guess) of makkelijker als np arrays --> netwerk vindt allebei goed
def fit_eight(model, dirlist, parent_path, epochs, timeframe, label_encoder):  # Fit a network on 8 datafiles at a time.
    fit_list = []
    i = 0
    j = 0
    while i < 8:  # Load the 8 datafiles
        filename = parent_path + "/" + dirlist[i]

        data = load_data.read_prepro_file(filename)
        label = extract_label(filename)

        fit_list.append((data, label))
        i += 1

    for epoch in range(epochs):
        print("Starting epoch {}".format((epoch+1)))
        train_acc = keras.metrics.CategoricalAccuracy()
        for counter, (data, label) in enumerate(fit_list):
            print("Fitting on file {} containing task: {}".format((counter+1), label))
            # Creates windows of length 'timeframe' to fit the model on
            windows = create_windows(data, timeframe)
            x_train = []
            # Only needed for CascadeNet, not for the LSTM network
            for window in windows:
                mesh = load_data.create_meshes(window)
                x_train.append(tf.convert_to_tensor(mesh))

            # Encode the textual label to one-hot-encoding
            encoded_label = label_encoder.transform([label])
            # Create a list of tensors, each corresponding to an encoded label
            y_train = [tf.convert_to_tensor(encoded_label)] * len(x_train)

            # Fit the model on the data from this specific file
            train_batch(model, x_train, y_train, train_acc)





def train_dir(dirpath, epochs, input_shape, timeframe, num_classes, label_encoder, shuffle=True):  # Train a network on a given directory
    # model misschien erbuiten aanmaken/specificeren om het meer modular te maken
    model = CascadeNet(input_shape, timeframe, num_classes)
    dirnames = os.listdir(dirpath)  # get list of all filenames
    random.shuffle(dirnames)
    batches = len(dirnames) // 8  # int divide by eight to avoid float errors

    # Check if it is divisible by eight
    if not (len(dirnames) % 8) == 0:
        raise TypeError("Not divisible by eight")

    for i in range(1):  # Fit on 8 batches at a time                HIER IPV 1 WEER 'BATCHES' NEERZETTEN
        print("Fitting on files {}-{}....".format((i*8), ((i+1)*8)))
        # Fits the model on 8 files for the specified amount of epochs
        fit_eight(model, dirnames[i * 8:(i + 1) * 8], dirpath, epochs, timeframe, label_encoder)
        print("--------------------------------------------------------------")

    # SAVE THE MODEL AFTER TRAINING!!!!

    return model


def extract_label(filename):  # Extract the label out of a filename
    filename = filename.split("/")
    pattern = r'_\d'
    split = re.split(pattern, filename[-1])
    return split[0]


def prepro_cross_files():  # Preprocess all the files and save them
    dirnames = os.listdir(c_train_path)
    for dir in dirnames:
        data = load_data.read_data_file(c_train_path + "/" + dir)
        new_data = downsampling.downsample_matrix(data, 3)
        scaled_data = load_data.scale(new_data, StandardScaler(), timewise=True)  # Beg that this returns an nparray
        try:
            hfive = h5py.File(c_train_prepro_path + "/" + dir, 'w')
        except:
            raise Exception("You forgot to make a train_prepro path")  # Can maybe be automated idk
        hfive.create_dataset('dir', data=scaled_data)
        hfive.close()


def prepro_intra_files():  # Preprocess all the files and save them
    dirnames = os.listdir(i_train_path)  # Get all the original files
    for dir in dirnames:  # For each file
        data = load_data.read_data_file(i_train_path + "/" + dir)  # Load the data
        new_data = downsampling.downsample_matrix(data, 3)  # Downsample the data
        scaled_data = load_data.scale(new_data, StandardScaler(), timewise=True)  # Scale the data
        try:
            hfive = h5py.File(i_train_prepro_path + "/" + dir, 'w')
        except:
            raise Exception("You forgot to make a train_prepro path")  # Open a new h5 object
        hfive.create_dataset('dir', data=scaled_data)  # Enter data into the object
        hfive.close()  # Close (save) the object


# print("pre")
# prepro_cross_files()
# print("pri")
# prepro_intra_files()
# print("pro")

# test = load_data.read_data_file("Final Project data/Intra/train/rest_105923_1.h5")
# print(test.shape)
# test_prepro = load_data.read_prepro_file("Final Project data/Intra/train_prepro/rest_105923_1.h5")
# print(test_prepro.shape)

textual_labels = ["rest", "task_motor", "task_story_math", "task_working_memory"]
label_encoder = LabelBinarizer()
label_encoder.fit(textual_labels)

train_dir("Final Project data/Intra/train_prepro", 1, cnn_input_shape, timeframe, output_classes, label_encoder)
# model = CascadeNet((20, 21, 1), 5, 4)
# model.compile(
#           loss=keras.losses.CategoricalCrossentropy(),
#           metrics=keras.metrics.CategoricalAccuracy(),
#           optimizer=keras.optimizers.Adam())
#
# data = load_data.read_prepro_file("Final Project data/Intra/train_prepro/rest_105923_1.h5")
# windows = create_windows(data, 5)
# meshes = []
# #mesh = np.expand_dims(load_data.create_meshes(windows[0]), axis=0)
# for window in windows:
#     mesh = load_data.create_meshes(window)
#     meshes.append(tf.convert_to_tensor(mesh))
#
# #x_train = np.stack(meshes, axis=0)
# x_train = meshes
# y_train = [tf.convert_to_tensor(label_encoder.transform(["rest"]))] * len(meshes)
# #y_train = [label_encoder.transform(["rest"])] * x_train.shape[0]
#
# train_acc = keras.metrics.CategoricalAccuracy()
#
#
# train_batch(model, x_train, y_train, train_acc)
