import tensorflow as tf
from tensorflow import keras
import os
import data_preprocessing
from training import extract_label
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(model_type, model_path, val_data_paths, timeframe, show_results=True):
    # Deze functie heette eerst validation omdat ik hem schreef met als doel validation, maar hij werkt wel als algemene evaluation functie
    # Voor validation kunnen we een deel van deze functie kopiëren en dan in een losse validate zetten en dit alleen voor echte evaluation gebruiken
    # (bijv. de confusion matrix plotten hoef niet voor validation set)
    model = keras.models.load_model(model_path)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.CategoricalCrossentropy())

    textual_labels = ["rest", "task_motor", "task_story_math", "task_working_memory"]
    label_encoder = LabelBinarizer()
    label_encoder.fit(textual_labels)

    x_val, y_val = get_test_set(model_type, val_data_paths, label_encoder, timeframe)

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


def get_test_set(model_type, test_data_paths, label_encoder, timeframe):
    all_data_files = []
    for dir_path in test_data_paths:
        dirnames = os.listdir(dir_path)
        for filename in dirnames:
            all_data_files.append(dir_path + '/' + filename)

    i = 0
    fit_list = []

    while i < len(all_data_files):  # Load the 16 datafiles
        filename = all_data_files[i]

        data = data_preprocessing.read_prepro_file(filename)
        label = extract_label(filename)

        fit_list.append((data, label))
        i += 1

    x_test = []
    y_test = []

    for counter, (data, label) in enumerate(fit_list):
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


model_path = "trained_models/trained_LSTM_e4_20_20_cross"
val_data_paths = ["MEG_Data/Final Project data/Cross/test1", "MEG_Data/Final Project data/Cross/test1"]
# BELANGRIJK OM AAN TE PASSEN AAN WAT HET MODEL TIJDENS TRAINEN HAD ANDERS GAAT HIJ HUILEN
timeframe = 20

evaluate_model("lstm", model_path, val_data_paths, timeframe)
