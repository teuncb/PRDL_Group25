import tensorflow as tf
from tensorflow import keras
import re
import os
import random
import numpy as np

import data_preprocessing
from cascadeNet import CascadeNet


# tf function zal niet heel veel uitmaken volgens de documentatie, omdat wij veel convolutional operations hebben (waarbij de speedup dus meevalt)
def train_epoch(model, x_train, y_train, optimizer, train_acc):
    # Instantiate an optimizer and loss function
    loss_function = keras.losses.CategoricalCrossentropy()

    total_loss = 0

    # Doet nu nog wel een update per window, wellicht veel? We kunnen ook nog batches gaan gebruiken om het aantal updates te reduceren
    for step in range(len(x_train)):
        # Used to track the gradients during the forward pass
        with tf.GradientTape() as tape:
            logits = model(tf.expand_dims(x_train[step], axis=0), training=True)
            # Calculate the loss value
            loss_value = loss_function(y_train[step], logits)

        total_loss += loss_value

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
    print("---------------------------------------------------")
    average_loss = total_loss / len(x_train)
    return average_loss, train_acc


# TODO: kijken of we de input als tensors willen doen (wat wel netter is I guess) of makkelijker als np arrays --> netwerk vindt allebei goed
def fit_eight(model, model_type, dirlist, parent_path, optimizer, timeframe, label_encoder,
              train_acc_obj):  # Fit a network on 8 datafiles at a time.
    fit_list = []
    average_loss = 0
    i = 0
    j = 0
    while i < 8:  # Load the 8 datafiles
        filename = parent_path + "/" + dirlist[i]

        data = data_preprocessing.read_prepro_file(filename)
        label = extract_label(filename)

        fit_list.append((data, label))
        i += 1

    for counter, (data, label) in enumerate(fit_list):
        print("Fitting on file {} containing task: {}".format((counter + 1), label))
        # Creates windows of length 'timeframe' to fit the model on
        windows = data_preprocessing.create_windows(data, model_type, timeframe)
        x_train = []
        # Only needed for CascadeNet, not for the LSTM network
        for window in windows:
            if model_type == "cascade":
                x_train.append(tf.convert_to_tensor(window))
            else:
                x_train.append(tf.convert_to_tensor(np.transpose(window)))

        # Encode the textual label to one-hot-encoding
        encoded_label = label_encoder.transform([label])
        # Create a list of tensors, each corresponding to an encoded label
        y_train = [tf.convert_to_tensor(encoded_label)] * len(x_train)

        # Fit the model on the data from this specific file
        average_loss, train_acc_obj = train_epoch(model, x_train, y_train, optimizer, train_acc_obj)

    return average_loss, train_acc_obj


def train_dir(model, model_type, dirpath, epochs, timeframe, optimizer, label_encoder,
              shuffle=True):  # Train a network on a given directory
    dirnames = os.listdir(dirpath)  # get list of all filenames
    random.shuffle(dirnames)
    batches = len(dirnames) // 8  # int divide by eight to avoid float errors

    train_acc_obj = keras.metrics.CategoricalAccuracy()

    average_epoch_accuracy = 0
    average_epoch_loss = 0
    losses = []
    accuracies = []

    # Check if it is divisible by eight
    if not (len(dirnames) % 8) == 0:
        raise TypeError("Not divisible by eight")

    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch + 1))
        print()
        train_acc_obj.reset_states()
        for i in range(batches):  # Fit on 8 batches at a time                HIER IPV 1 WEER 'BATCHES' NEERZETTEN
            print("Fitting on files {}-{}....".format((i * 8), ((i + 1) * 8)))
            # Fits the model on 8 files for the specified amount of epochs
            average_loss, train_acc_obj = fit_eight(model, model_type, dirnames[i * 8:(i + 1) * 8], dirpath, optimizer,
                                                    timeframe, label_encoder, train_acc_obj)
            average_epoch_loss += average_loss
            average_epoch_accuracy += train_acc_obj.result()

            print("--------------------------------------------------------------")

        losses.append(average_epoch_loss)
        accuracies.append(average_epoch_accuracy)

    # SAVE THE MODEL AFTER TRAINING!!!!

    return model, losses, accuracies


def extract_label(filename):  # Extract the label out of a filename
    filename = filename.split("/")
    pattern = r'_\d'
    split = re.split(pattern, filename[-1])
    return split[0]
