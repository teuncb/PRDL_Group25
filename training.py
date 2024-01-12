import tensorflow as tf
from tensorflow import keras
import re
import os
import random

import data_preprocessing
from cascadeNet import CascadeNet


# tf function zal niet heel veel uitmaken volgens de documentatie, omdat wij veel convolutional operations hebben (waarbij de speedup dus meevalt)
def train_batch(model, x_train, y_train, train_acc):
    # Instantiate an optimizer and loss function
    optimizer = keras.optimizers.Adam()
    loss_function = keras.losses.CategoricalCrossentropy()

    # Doet nu nog wel een update per window, wellicht veel? We kunnen ook nog batches gaan gebruiken om het aantal updates te reduceren
    for step in range(len(x_train)):
        # Used to track the gradients during the forward pass
        with tf.GradientTape() as tape:
            logits = model(tf.expand_dims(x_train[step], axis=0), training=True)
            # Calculate the loss value
            loss_value = loss_function(y_train[step], logits)

        # Extract the gradients from the gradienttape
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Perform a weight update step using the extracted gradients
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # Keep track of the training accuracy which is given at the end of the loop
        train_acc.update_state(y_train[step], logits)

        # Report the training loss for monitoring
        if step % 100 == 0:
            print("Training loss value at step {}: {}".format(step, loss_value))

    print("Training accuracy over whole file: {}".format(train_acc.result()))


# TODO: kijken of we de input als tensors willen doen (wat wel netter is I guess) of makkelijker als np arrays --> netwerk vindt allebei goed
def fit_eight(model, dirlist, parent_path, epochs, timeframe, label_encoder):  # Fit a network on 8 datafiles at a time.
    fit_list = []
    i = 0
    j = 0
    while i < 8:  # Load the 8 datafiles
        filename = parent_path + "/" + dirlist[i]

        data = data_preprocessing.read_prepro_file(filename)
        label = extract_label(filename)

        fit_list.append((data, label))
        i += 1

    for epoch in range(epochs):
        print("Starting epoch {}".format((epoch+1)))
        train_acc = keras.metrics.CategoricalAccuracy()
        for counter, (data, label) in enumerate(fit_list):
            print("Fitting on file {} containing task: {}".format((counter+1), label))
            # Creates windows of length 'timeframe' to fit the model on
            windows = data_preprocessing.create_windows(data, timeframe)
            x_train = []
            # Only needed for CascadeNet, not for the LSTM network
            for window in windows:
                mesh = data_preprocessing.create_meshes(window)
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