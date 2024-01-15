import tensorflow as tf
from tensorflow import keras
import os
import data_preprocessing
from training import extract_label
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def validate_LSTM(model_path, val_data_path, timeframe):
    # Deze functie heet nog validation omdat ik hem schreef met als doel validation, maar hij werkt wel als algemene evaluation functie
    # Voor validation kunnen we een deel van deze functie kopiëren en dan in een losse validate zetten en dit alleen voor echte evaluation gebruiken
    # (bijv. de confusion matrix plotten hoef niet voor validation set)
    model = keras.models.load_model(model_path)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.CategoricalCrossentropy())

    textual_labels = ["rest", "task_motor", "task_story_math", "task_working_memory"]
    label_encoder = LabelBinarizer()
    label_encoder.fit(textual_labels)

    print("Getting validation set")
    x_val, y_val = get_test_set("lstm", val_data_path, label_encoder, timeframe)
    print("Validation set acquired")

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

    # conf_matrix = confusion_matrix(numeric_y_true, predictions)
    #
    # # Ik had eerst de textuele classnames erin gezet, maar doordat die lang zijn wordt dat best lelijk
    # # --> even kijken hoe we dat netjes op kunnen lossen
    # disp = ConfusionMatrixDisplay(conf_matrix)
    # disp.plot()
    # plt.tight_layout()
    # plt.show()

def validate_cascadeNet(model_path, val_data_path, timeframe):
    # Deze functie heet nog validation omdat ik hem schreef met als doel validation, maar hij werkt wel als algemene evaluation functie
    # Voor validation kunnen we een deel van deze functie kopiëren en dan in een losse validate zetten en dit alleen voor echte evaluation gebruiken
    # (bijv. de confusion matrix plotten hoef niet voor validation set)
    model = keras.models.load_model(model_path)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.CategoricalCrossentropy())

    textual_labels = ["rest", "task_motor", "task_story_math", "task_working_memory"]
    label_encoder = LabelBinarizer()
    label_encoder.fit(textual_labels)

    print("Getting validation set")
    x_val, y_val = get_test_set("cascade", val_data_path, label_encoder, timeframe)
    print("Validation set acquired")
    print(len(x_val))
    print(len(y_val))

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

    # conf_matrix = confusion_matrix(numeric_y_true, predictions)
    #
    # # Ik had eerst de textuele classnames erin gezet, maar doordat die lang zijn wordt dat best lelijk
    # # --> even kijken hoe we dat netjes op kunnen lossen
    # disp = ConfusionMatrixDisplay(conf_matrix)
    # disp.plot()
    # plt.tight_layout()
    # plt.show()


def get_test_set(model_type, test_data_path, label_encoder, timeframe):
    dirnames = os.listdir(test_data_path)
    i = 0
    fit_list = []

    while i < len(dirnames):
        filename = test_data_path + "/" + dirnames[i]

        data = data_preprocessing.read_prepro_file(filename)
        label = extract_label(filename)

        fit_list.append((data, label))
        i += 1

    x_test = []
    y_test = []

    for counter, (data, label) in enumerate(fit_list):
        if model_type == "cascade":
            meshes = data_preprocessing.create_meshes(data)
            windows = data_preprocessing.create_windows(meshes, model_type, timeframe)
        else:
            # Creates windows of length 'timeframe' to fit the model on
            windows = data_preprocessing.create_windows(data, model_type, timeframe)

        # Encode the textual label to one-hot-encoding
        encoded_label = label_encoder.transform([label])


        # Only needed for CascadeNet, not for the LSTM network
        for window in windows:
            # Create a list of tensors, each corresponding to an encoded label
            y_test.append(tf.convert_to_tensor(encoded_label))
            if model_type == "cascade":
                x_test.append(tf.convert_to_tensor(window))
            else:
                x_test.append(tf.convert_to_tensor(np.transpose(window)))

    return x_test, y_test

def validation():
    CascadeNet_path = "Trained CascadeNet/trained_CascadeNet_intra"
    LSTM_path = "Trained LSTMs/trained_LSTM_e3_5_5_intra"
    val_data_path = "Final Project data/Validation"

    #validate_LSTM(LSTM_path, val_data_path, 5)
    intra_path = "Trained LSTMs/Final Models/trained_LSTM_e{}_{}_{}_intra"
    cross_path = "Trained LSTMs/Final Models/trained_LSTM_e{}_{}_{}_cross"

    learning_rates = [2, 3, 4]
    timeframes = [20]
    units = [2, 5, 20]


    for lr in learning_rates:
        for frame in timeframes:
            for unit in units:
                model_path = cross_path.format(lr, frame, unit)
                print("Score for: {}".format(model_path))
                validate_LSTM(model_path, val_data_path, frame)
                print("----------------------------------")


def evaluate_model(model_type, model_path, test_paths, timeframe, show_results=True):
    # Deze functie heette eerst validation omdat ik hem schreef met als doel validation, maar hij werkt wel als algemene evaluation functie
    # Voor validation kunnen we een deel van deze functie kopiëren en dan in een losse validate zetten en dit alleen voor echte evaluation gebruiken
    # (bijv. de confusion matrix plotten hoef niet voor validation set)
    model = keras.models.load_model(model_path)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.CategoricalCrossentropy())

    textual_labels = ["rest", "task_motor", "task_story_math", "task_working_memory"]
    label_encoder = LabelBinarizer()
    label_encoder.fit(textual_labels)

    x_val, y_val = get_test_set(model_type, test_paths, label_encoder, timeframe)

    # Generate logit predictions by the model on the validation data
    logits = model(x_val)
    # Convert the logits to actual predicted classes
    predictions = [np.argmax(logit) for logit in logits]
    # Convert the one-hot-encoded y_true labels to a numeric form
    numeric_y_true = [np.argmax(y) for y in y_val]

    # Calculate performance metrics
    accuracy = accuracy_score(numeric_y_true, predictions)
    if show_results:
        print("Accuracy = {}".format(accuracy))

    performance_metrics = precision_recall_fscore_support(numeric_y_true, predictions, average="macro")
    if show_results:
        print("Precision = {}, recall = {}, F1-score = {}".format(performance_metrics[0], performance_metrics[1],
                                                              performance_metrics[2]))

    if show_results:
        conf_matrix = confusion_matrix(numeric_y_true, predictions)

        # Ik had eerst de textuele classnames erin gezet, maar doordat die lang zijn wordt dat best lelijk
        # --> even kijken hoe we dat netjes op kunnen lossen
        disp = ConfusionMatrixDisplay(conf_matrix)
        disp.plot()
        plt.tight_layout()
        plt.show()

    return accuracy

test_paths = ["Final Project data/Cross/test1", "Final Project data/Cross/test1"]
model_path = "Trained LSTMs/trained_LSTM_e4_20_20_intra"

evaluate_model("lstm", model_path, test_paths, 20)