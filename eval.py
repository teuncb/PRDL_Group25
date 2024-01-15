import tensorflow as tf
from tensorflow import keras
import os
import data_preprocessing
from training import extract_label
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.engine.input_spec import InputSpec


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
    x_val, y_val = get_val_set("lstm", val_data_path, label_encoder, timeframe)
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

    conf_matrix = confusion_matrix(numeric_y_true, predictions)

    # Ik had eerst de textuele classnames erin gezet, maar doordat die lang zijn wordt dat best lelijk
    # --> even kijken hoe we dat netjes op kunnen lossen
    disp = ConfusionMatrixDisplay(conf_matrix)
    disp.plot()
    plt.tight_layout()
    plt.show()

def validate_convLSTM(model_path, val_data_path, timeframe):
    # Deze functie heet nog validation omdat ik hem schreef met als doel validation, maar hij werkt wel als algemene evaluation functie
    # Voor validation kunnen we een deel van deze functie kopiëren en dan in een losse validate zetten en dit alleen voor echte evaluation gebruiken
    # (bijv. de confusion matrix plotten hoef niet voor validation set)
    model = keras.models.load_model(model_path, compile=False)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.CategoricalCrossentropy())

    textual_labels = ["rest", "task_motor", "task_story_math", "task_working_memory"]
    label_encoder = LabelBinarizer()
    label_encoder.fit(textual_labels)

    print("Getting validation set")
    x_val, y_val = get_test_set("cascade", val_data_path, label_encoder, timeframe)
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

    conf_matrix = confusion_matrix(numeric_y_true, predictions)

    # Ik had eerst de textuele classnames erin gezet, maar doordat die lang zijn wordt dat best lelijk
    # --> even kijken hoe we dat netjes op kunnen lossen
    disp = ConfusionMatrixDisplay(conf_matrix)
    disp.plot()
    plt.tight_layout()
    plt.show()


def get_val_set(model_type, val_data_path, label_encoder, timeframe):
    dirnames = os.listdir(val_data_path)
    i = 0
    fit_list = []

    while i < len(dirnames):
        filename = val_data_path + "/" + dirnames[i]

        data = data_preprocessing.read_prepro_file(filename)
        label = extract_label(filename)

        fit_list.append((data, label))
        i += 1

    x_val = []
    y_val = []

    for counter, (data, label) in enumerate(fit_list):
        if model_type == "cascade":
            meshes = data_preprocessing.create_meshes(data)
            windows = data_preprocessing.create_windows(meshes, model_type, timeframe)
        else:
            # Creates windows of length 'timeframe' to fit the model on
            windows = data_preprocessing.create_windows(data, model_type, timeframe)

        # Encode the textual label to one-hot-encoding
        encoded_label = label_encoder.transform([label])


        # Only needed for ConvLSTM, not for the LSTM network
        for window in windows:
            # Create a list of tensors, each corresponding to an encoded label
            y_val.append(tf.convert_to_tensor(encoded_label))
            if model_type == "cascade":
                res = np.reshape(window, (timeframe, 20, 21))
                x_val.append(tf.convert_to_tensor(np.expand_dims(res, axis=-1)))
            else:
                x_val.append(tf.convert_to_tensor(np.transpose(window)))

    return x_val, y_val

def validation(task):
    CascadeNet_path = "Trained CascadeNet/trained_CascadeNet_intra"
    LSTM_path = "Trained LSTMs/trained_LSTM_e3_5_5_intra"
    if task == "cross":
        val_data_path = "Final Project data/Validation"
        path = "Trained LSTMs/Final Models/trained_LSTM_e{}_{}_{}_cross"
    else:
        val_data_path = "Final Project data/Intra/val_prepro"
        path = "Trained LSTMs/Final Models/trained_LSTM_e{}_{}_{}_intra"

    learning_rates = [4]
    timeframes = [5]
    units = [5, 20]


    for lr in learning_rates:
        for frame in timeframes:
            for unit in units:
                model_path = path.format(lr, frame, unit)
                print("Score for: {}".format(model_path))
                validate_LSTM(model_path, val_data_path, frame)
                print("----------------------------------")



def get_test_set(model_type, test_data_path, label_encoder, timeframe):
    # Note that 'test_data_path' needs to be a list, even if it consists of one path (such as in Intra)
    filenames = []
    parent_paths = []
    for dir in test_data_path:
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

    x_test = []
    y_test = []

    for counter, (data, label) in enumerate(test_list):
        if model_type == "convlstm":
            meshes = data_preprocessing.create_meshes(data)
            windows = data_preprocessing.create_windows(meshes, model_type, timeframe)
        else:
            # Creates windows of length 'timeframe' to fit the model on
            windows = data_preprocessing.create_windows(data, model_type, timeframe)

        # Encode the textual label to one-hot-encoding
        encoded_label = label_encoder.transform([label])


        # Only needed for ConvLSTM, not for the LSTM network
        for window in windows:
            # Create a list of tensors, each corresponding to an encoded label
            y_test.append(tf.convert_to_tensor(encoded_label))
            if model_type == "convlstm":
                res = np.reshape(window, (timeframe, 20, 21))
                x_test.append(tf.convert_to_tensor(np.expand_dims(res, axis=-1)))
            else:
                x_test.append(tf.convert_to_tensor(np.transpose(window)))

    return x_test, y_test



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


# Negeer ff al deze dingen en comments, is voor al het testen en aangezien ik dit nog wel vaker moet doen
# laat ik het liever nog even staan. Maar bij deze wel alvast een werkende eval
test_paths = ["Final Project data/Cross/test1_prepro", "Final Project data/Cross/test2_prepro"]
#test_paths = ["Final Project data/Intra/test_prepro"]
model_path = "Trained LSTMs/trained_LSTM_e4_20_20_intra_dropoutentripleentienepochs"

#evaluate_model("lstm", model_path, test_paths, 20)

#validation("cross")

# LSTM_path = "Trained LSTMs/trained_LSTM_e4_20_20_intra_dropoutentripleentienepochs"
# val_data_path = "Final Project data/Intra/val_prepro"
# #
# validate_LSTM(LSTM_path, val_data_path, 20)
# convLSTM = "Trained ConvLSTM/trained_ConvLSTM_intra_epoch1"
# validate_convLSTM(convLSTM, val_data_path, 20)

textual_labels = ["rest", "task_motor", "task_story_math", "task_working_memory"]
label_encoder = LabelBinarizer()
label_encoder.fit(textual_labels)

evaluate_model("lstm", model_path, test_paths, 20)

