import tensorflow as tf
from tensorflow import keras
import os
import data_preprocessing
from OLD_training import extract_label
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def validate_model(model, model_type, val_data_path, timeframe):
    """
    Given a model and the path to the validation set, evaluates the model on this validation data.

    :param model: the trained Tensorflow model that needs to be evaluated.
    :param model_type: whether the model is of type 'lstm' or 'convlstm'. This changed which input shape is needed.
    :param val_data_path: the path to the validation data.
    :param timeframe: how many timesteps are included in one sample, needed for splitting the data into windows.
    :return: Void.
    """
    # Create a label encoder and fit it on our labels
    textual_labels = ["rest", "task_motor", "task_story_math", "task_working_memory"]
    label_encoder = LabelBinarizer()
    label_encoder.fit(textual_labels)

    # Load the validation data and preprocess it in the right way for the provided model type
    x_val, y_val = get_val_set(model_type, val_data_path, label_encoder, timeframe)

    print("Generating predictions")
    # Generate logit predictions by the model on the validation data
    logits = model(x_val, training=False)
    print("Predictions generated")
    # Convert the logits to actual predicted classes
    predictions = [np.argmax(logit) for logit in logits]
    # Convert the one-hot-encoded y_true labels to a numeric form
    numeric_y_true = [np.argmax(y) for y in y_val]

    # Calculate performance metrics
    accuracy = accuracy_score(numeric_y_true, predictions)
    print("Accuracy = {}".format(accuracy))

    performance_metrics = precision_recall_fscore_support(numeric_y_true, predictions, average="macro")
    print("Precision = {}, recall = {}, F1-score = {}".format(performance_metrics[0], performance_metrics[1],
                                                              performance_metrics[2]))

    conf_matrix = confusion_matrix(numeric_y_true, predictions)

    disp = ConfusionMatrixDisplay(conf_matrix)
    disp.plot()
    plt.tight_layout()
    plt.show()

def get_val_set(model_type, val_data_path, label_encoder, timeframe):
    """
    Load and preprocess the validation data in the right format for the model type.

    :param model_type: whether the data should be converted to meshes (in case of 'convlstm').
    :param val_data_path: the place from where to load the validation data.
    :param label_encoder: a trained label encoder, used to transform the textual labels into numbers.
    :param timeframe: how many timesteps are in one window.
    :return: validation training samples (x_val) and their labels (y_val)
    """
    # Get all files in the validation folder
    dirnames = os.listdir(val_data_path)
    i = 0
    fit_list = []

    # Load all data from the files and extract their label
    while i < len(dirnames):
        filename = val_data_path + "/" + dirnames[i]

        data = data_preprocessing.read_prepro_file(filename)
        label = extract_label(filename)

        fit_list.append((data, label))
        i += 1

    # Will contain all windows (represented as arrays), which are later stacked to form x_val and y_val
    x_temp = []
    y_temp = []

    # Loop through the files
    for counter, (data, label) in enumerate(fit_list):
        if model_type == "convlstm":
            # First convert the data to 2D meshes before dividing it into windows
            meshes = data_preprocessing.create_meshes(data)
            windows = data_preprocessing.create_windows(meshes, model_type, timeframe)
        else:
            # Creates windows of length 'timeframe' to fit the model on
            windows = data_preprocessing.create_windows(data, model_type, timeframe)

        # Will be a temporary savepoint for the windows
        x_windows = []

        for window in windows:
            if model_type == "convlstm":
                # The data comes out as (20, 21, timeframe), the ConvLSTM expect timeframe before width/height
                res = np.reshape(window, (timeframe, 20, 21))
                x_windows.append(np.expand_dims(res, axis=-1))
            else:
                x_windows.append(np.transpose(window))

        # Add all the generated windows of this file to x_temp
        x_temp.extend(x_windows)

        # Encode the textual label to one-hot-encoding
        encoded_label = label_encoder.transform([label])
        y_current = [np.squeeze(encoded_label)] * len(x_windows)
        y_temp.extend(y_current)

    # Stack the lists of arrays x_temp, y_temp in the first dimension, which is the batch size
    x_val = np.stack(x_temp, axis=0)
    y_val = np.stack(y_temp, axis=0)
    return x_val, y_val

def get_test_set(model_type, test_data_paths, label_encoder, timeframe):
    """
    Essentially the same function as get_val_set(), except for the fact that you can give a list of test paths.

    :param model_type: whether the data should be converted to meshes (in case of 'convlstm').
    :param test_data_path: the place from where to load the test data.
    :param label_encoder: a trained label encoder, used to transform the textual labels into numbers.
    :param timeframe: how many timesteps are in one window.
    :return: test training samples (x_test) and their labels (y_test)
    """
    # Note that 'test_data_path' needs to be a list, even if it consists of one path (such as in Intra)
    filenames = []
    parent_paths = []
    # Loop through the paths included in test_data_paths and load the data
    for dir in test_data_paths:
        files = os.listdir(dir)
        filenames.extend(files)
        parent_paths.extend([dir] * len(files))

    test_list = []

    i = 0
    for filename in filenames:
        data = data_preprocessing.read_prepro_file(parent_paths[i] + "/" + filename)
        label = extract_label(filename)

        test_list.append((data, label))
        i += 1

    x_temp = []
    y_temp = []

    for counter, (data, label) in enumerate(test_list):
        if model_type == "convlstm":
            # First convert the loaded data into 2D meshes
            meshes = data_preprocessing.create_meshes(data)
            windows = data_preprocessing.create_windows(meshes, model_type, timeframe)
        else:
            # Creates windows of length 'timeframe' to fit the model on
            windows = data_preprocessing.create_windows(data, model_type, timeframe)

        x_windows = []

        for window in windows:
            if model_type == "convlstm":
                # Just like with the validation data, we need to reshape the windows first
                res = np.reshape(window, (timeframe, 20, 21))
                x_windows.append(np.expand_dims(res, axis=-1))
            else:
                x_windows.append(np.transpose(window))

        x_temp.extend(x_windows)

        # Encode the textual label to one-hot-encoding
        encoded_label = label_encoder.transform([label])
        y_current = [np.squeeze(encoded_label)] * len(x_windows)
        y_temp.extend(y_current)

    # Stack the arrays included in the lists x_temp / y_temp to form a batch of test data
    x_test = np.stack(x_temp, axis=0)
    y_test = np.stack(y_temp, axis=0)

    return x_test, y_test



def evaluate_model(model_type, model, test_paths, timeframe):
    """
    Uses the provided model to generate predictions on the test data, and evaluates these predictions.

    :param model_type: whether the model is an 'lstm' or 'convlstm'.
    :param model: the model that needs to be evaluated.
    :param test_paths: list of paths to the test data.
    :param timeframe: how many timesteps are in one window.
    :return: the accuracy score of the model on the test set.
    """

    textual_labels = ["rest", "task_motor", "task_story_math", "task_working_memory"]
    label_encoder = LabelBinarizer()
    label_encoder.fit(textual_labels)

    # Load and preprocess the test data
    x_test, y_test = get_test_set(model_type, test_paths, label_encoder, timeframe)

    # Generate logit predictions by the model on the validation data
    logits = model(x_test)

    # Convert the logits to actual predicted classes
    predictions = [np.argmax(logit) for logit in logits]
    # Convert the one-hot-encoded y_true labels to a numeric form
    numeric_y_true = [np.argmax(y) for y in y_test]

    print("Predictions: {}".format(predictions))
    print("True Y: {}".format(numeric_y_true))

    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(numeric_y_true, predictions)
    print("Accuracy = {}".format(accuracy))

    # Calculates the performance metrics averaged over all classes
    average_performance_metrics = precision_recall_fscore_support(numeric_y_true, predictions, average="macro")
    print("Precision = {}, recall = {}, F1-score = {}".format(average_performance_metrics[0], average_performance_metrics[1],
                                                          average_performance_metrics[2]))

    # Calculates the performance metrics for each class, giving more insight into how the model performs on the
    # different classes
    class_specific_metrics = precision_recall_fscore_support(numeric_y_true, predictions)
    print("Precision = {}, recall = {}, F1-score = {}".format(class_specific_metrics[0], class_specific_metrics[1],
                                                              class_specific_metrics[2]))

    # Generate a confusion matrix using the predictions and true labels
    conf_matrix = confusion_matrix(numeric_y_true, predictions)

    # Plot the generated confusion matrix
    disp = ConfusionMatrixDisplay(conf_matrix)
    disp.plot()
    plt.tight_layout()
    plt.show()

    return accuracy
