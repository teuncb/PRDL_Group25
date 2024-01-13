import tensorflow as tf
import keras
import os
import data_preprocessing
from training import extract_label

def evaluate(model_path, test_data_path):
    model = keras.models.load_model(model_path)
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=keras.metrics.CategoricalAccuracy(),
        optimizer=keras.optimizers.Adam()
    )

    x_test, y_test = get_test_set(test_data_path)

    results = model.evaluate(x_test, y_test)
    print(results)


def get_test_set(test_data_path):
    dirnames = os.listdir(test_data_path)

    data_list = []
    label_list = []
    i = 0
    while i < 16:
        filename = test_data_path + "/" + dirnames[i]

        data = data_preprocessing.read_prepro_file(filename)
        label = extract_label(filename)

        data_list.append(data)
        label_list.append(label)
        i += 1

    print(data_list, label_list)

    x_test = tf.convert_to_tensor(data_list)
    y_test = tf.convert_to_tensor(label_list)

    return x_test, y_test

# test
evaluate("trained_models/trained_LSTM_e2_20_2_cross", "MEG_Data/Final Project data/Cross/test1_prepro")