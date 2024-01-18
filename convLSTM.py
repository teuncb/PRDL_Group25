import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
import data_preprocessing
from OLD_training import train_file


class ConvLSTM(keras.Model):
    def __init__(self, input_shape, timeframe, lstm_units, num_classes=4):
        super().__init__()

        self.in_shape = input_shape
        self.timeframe = timeframe
        self.lstm_units = lstm_units
        self.num_classes = num_classes

        self.conv_lstm1 = keras.layers.ConvLSTM2D(filters=8, kernel_size=(3, 3), padding="same",
                                                  input_shape=self.in_shape, return_sequences=False, data_format="channels_last")

        self.conv_lstm2 = keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), padding="same")

        self.flat = keras.layers.Flatten()

        self.out = keras.layers.Dense(self.num_classes, activation="softmax")

    def call(self, inputs, training=True, mask=None):
        """
        Specifies how the inputs should be passed through the layers of the model and returns the output

        :param inputs: the inputs to be classified by the model, with the same shape as self.in_shape
        :param training: whether the gradients should be tracked
        :param mask: whether a certain mask should be applied on the inputs (such as masking certain timesteps)
        :return: probabilities for each of the classes
        """

        input_layer = keras.layers.InputLayer(self.in_shape, name="input")(inputs)

        conv_lstm1 = self.conv_lstm1(input_layer)

        flat = self.flat(conv_lstm1)

        # Final fully connected layer with softmax to give class probabilities
        output = self.out(flat)

        return output

    def build_graph(self):
        """
        Builds the Tensor Graph in order to generate a summary of the model by running dummy input through the model
        :return: a 'dummy' version of the model of which we can generate a summary
        """
        x = keras.Input(self.in_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))
