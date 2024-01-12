import tensorflow as tf
import numpy as np
from tensorflow import keras
import logging
import data_preprocessing
from sklearn.preprocessing import LabelBinarizer
from training import train_batch


class JustLSTM(keras.Model):
    def __init__(self, input_shape, timeframe, num_classes=4):
        super().__init__()

        # TODO: kijken of we input shape niet uit elkaar trekken in width, height, batch_size en timesteps (nu num_segments)
        self.in_shape = input_shape
        self.num_classes = num_classes

        # The number of images / timesteps that we will look at for each training step
        self.timeframe = timeframe

        self.lstm1 = keras.layers.LSTM(self.timeframe, return_sequences=True, name="lstm1")
        self.lstm2 = keras.layers.LSTM(self.timeframe, name="lstm2")

        self.fc = keras.layers.Dense(32, activation="relu")

        self.out = keras.layers.Dense(self.num_classes, activation="softmax", name="output")

    def call(self, inputs, training=True, mask=None):
        """
        Specifies how the inputs should be passed through the layers of the model and returns the output

        :param inputs: the inputs to be classified by the model, with the same shape as self.in_shape
        :param training: whether the gradients should be tracked
        :param mask: whether a certain mask should be applied on the inputs (such as masking certain timesteps)
        :return: probabilities for each of the classes
        """

        input_layer = keras.layers.InputLayer(self.in_shape, name="input")(inputs)

        # Pass the timesteps through the RNN to find temporal features The amount of units in the layer are equal to
        # the number of timesteps (i.e. segments) according to Zhang et al. (2018)
        lstm1 = self.lstm1(input_layer)
        lstm2 = self.lstm2(lstm1)

        fc = self.fc(lstm2)

        # Final fully connected layer with softmax to give class probabilities
        output = self.out(fc)

        return output

    def build_graph(self):
        """
        Builds the Tensor Graph in order to generate a summary of the model by running dummy input through the model
        :return: a 'dummy' version of the model of which we can generate a summary
        """
        x = keras.Input(self.in_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))


# Quick test
model = JustLSTM((5, 248), 5)

textual_labels = ["rest", "task_motor", "task_story_math", "task_working_memory"]
label_encoder = LabelBinarizer()
label_encoder.fit(textual_labels)

data = data_preprocessing.read_prepro_file("Final Project data/Intra/train_prepro/rest_105923_1.h5")
windows = data_preprocessing.create_windows(data, 5)

x_train = []

for window in windows:
    x_train.append(tf.convert_to_tensor(np.transpose(window)))

y_train = [tf.convert_to_tensor(label_encoder.transform(["rest"]))] * len(x_train)

train_acc = keras.metrics.CategoricalAccuracy()

train_batch(model, x_train, y_train, train_acc)
