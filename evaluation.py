import tensorflow as tf
from tensorflow import keras
import os
import data_preprocessing
from training import extract_label
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def validation(model_path, val_data_path, timeframe):
    # Deze functie heet nog validation omdat ik hem schreef met als doel validation, maar hij werkt wel als algemene evaluation functie
    # Voor validation kunnen we een deel van deze functie kopiëren en dan in een losse validate zetten en dit alleen voor echte evaluation gebruiken
    # (bijv. de confusion matrix plotten hoef niet voor validation set)
    model = keras.models.load_model(model_path)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.CategoricalCrossentropy())

    textual_labels = ["rest", "task_motor", "task_story_math", "task_working_memory"]
    label_encoder = LabelBinarizer()
    label_encoder.fit(textual_labels)

    x_val, y_val = get_test_set("lstm", val_data_path, label_encoder, timeframe)

    # Generate logit predictions by the model on the validation data
    logits = model(x_val)
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


def get_test_set(model_type, test_data_path, label_encoder, timeframe):
    dirnames = os.listdir(test_data_path)
    i = 0
    fit_list = []

    while i < len(dirnames):  # Load the 8 datafiles
        filename = test_data_path + "/" + dirnames[i]

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


model_path = "Trained LSTMs/trained_LSTM_e4_20_20_intra"
val_data_path = "Final Project data/Validation"
# BELANGRIJK OM AAN TE PASSEN AAN WAT HET MODEL TIJDENS TRAINEN HAD ANDERS GAAT HIJ HUILEN
timeframe = 20

validation(model_path, val_data_path, timeframe)
