import tensorflow as tf
import numpy as np
from tensorflow import keras
import logging
import data_preprocessing
from sklearn.preprocessing import LabelBinarizer
from OLD_training import train_file


class JustLSTM(keras.Model):
    def __init__(self, input_shape, timeframe, lstm_units, num_classes=4):
        super().__init__()

        self.in_shape = input_shape
        self.num_classes = num_classes

        # The number of images / timesteps that we will look at for each training step
        self.timeframe = timeframe
        self.lstm_units = lstm_units


        self.lstm1 = keras.layers.LSTM(self.lstm_units, return_sequences=True, name="lstm1", kernel_regularizer=keras.regularizers.l2(0.01))
        self.lstm2 = keras.layers.LSTM(self.lstm_units, name="lstm2", dropout=0.2, kernel_regularizer=keras.regularizers.l2(0.01))

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

        # Final fully connected layer with softmax to give class probabilities
        output = self.out(lstm2)

        return output

    def build_graph(self):
        """
        Builds the Tensor Graph in order to generate a summary of the model by running dummy input through the model
        :return: a 'dummy' version of the model of which we can generate a summary
        """
        x = keras.Input(self.in_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))
