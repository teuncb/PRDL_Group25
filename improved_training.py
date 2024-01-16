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
    Extract the label out of a filename

    :param filename: the filename that includes the label
    :return: the label
    """
    filename = filename.split("/")
    pattern = r'_\d'
    # Split the filename based on the regular expression
    split = re.split(pattern, filename[-1])
    # The label is the first item in the filename
    return split[0]


def train_dir(model, model_type, dirpath, epochs, timeframe, label_encoder):  # Train a network on a given directory
    dirnames = os.listdir(dirpath)  # get list of all filenames
    random.shuffle(dirnames)
    batches = len(dirnames) // 8  # int divide by eight to avoid float errors

    history = []
    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch + 1))
        print()
        for i in range(batches):  # Fit on 8 batches at a time
            fit_list = []
            print("Fitting on files {}-{}....".format((i * 8), ((i + 1) * 8)))
            # Fits the model on 8 files for the specified amount of epochs
            files = dirnames[i * 8:(i + 1) * 8]
            for j in range(8):
                filename = dirpath + "/" + files[j]

                data = data_preprocessing.read_prepro_file(filename)
                label = extract_label(filename)

                fit_list.append((data, label))


            x_temp = []
            y_temp = []
            for counter, (data, label) in enumerate(fit_list):
                #print("Fitting on file {} containing task: {}".format((counter + 1), label))
                # Creates windows of length 'timeframe' to fit the model on
                windows = data_preprocessing.create_windows(data, model_type, timeframe)
                x_windows = []
                # Only needed for CascadeNet, not for the LSTM network
                for window in windows:
                    if model_type == "convlstm":
                        x_windows.append(np.expand_dims(np.transpose(window), axis=-1))
                    else:
                        x_windows.append(np.transpose(window))
                x_temp.extend(x_windows)
                # Encode the textual label to one-hot-encoding
                encoded_label = label_encoder.transform([label])

                y_current = [np.squeeze(encoded_label)] * len(x_windows)
                y_temp.extend(y_current)

            x_train = np.stack(x_temp, axis=0)
            #print(x_train.shape)

            y_train = np.stack(y_temp, axis=0)
            #print(y_train.shape)


            hist = model.fit(x_train, y_train, epochs)

            print("--------------------------------------------------------------")

    # SAVE THE MODEL AFTER TRAINING!!!!
    #model.save("Single_Layer_LSTM_newtrain")

    return model, history


timeframe = 10
# model = JustLSTM((timeframe, 248), timeframe, 50, 10)
model = ConvLSTM((timeframe, 20, 21, 1), timeframe, 50)
model.compile(loss=keras.losses.CategoricalCrossentropy(),
              metrics=keras.metrics.CategoricalAccuracy(),
              optimizer=keras.optimizers.Adam(learning_rate=5e-5))

textual_labels = ["rest", "task_motor", "task_story_math", "task_working_memory"]
label_encoder = LabelBinarizer()
label_encoder.fit(textual_labels)

mod, history = train_dir(model, "convlstm", "Final Project data/Intra/train_prepro_mesh", 1, timeframe, label_encoder)

eval.validate_convLSTM(model, "Final Project data/Validation", timeframe)

#evaluate_model("convlstm", model, ["Final Project data/Intra/test_prepro"], 5)

