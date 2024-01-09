import tensorflow as tf
import numpy as np
import keras
import logging

class JustLSTM(keras.Model):
    def __init__(self, input_shape, num_segments, num_classes=4):
        super().__init__()

        # TODO: kijken of we input shape niet uit elkaar trekken in width, height, batch_size en timesteps (nu num_segments)
        self.in_shape = input_shape
        self.num_classes = num_classes

        # The number of images / timesteps that we will look at for each training step
        self.timeframe = num_segments

        # To get the data in the shape [batch, timesteps, features] for LSTM, we need to expand the dimensions
        # of the output of the fully connected layer that is of dimension [batch, 1024] to [batch, 1, 1024]
        self.lamb = keras.layers.Lambda(lambda previous: tf.expand_dims(previous, axis=1), name="expand_dims")

        # Concatenating the dense outputs of all timesteps (e.g. 5) gives the right input shape for the LSTM layer [batch, 5, 1024]
        self.concat = keras.layers.Concatenate(axis=1, name="concat")

        self.lstm = keras.layers.LSTM(self.timeframe, name="lstm")

        self.out = keras.layers.Dense(self.num_classes, activation="softmax", name="output")

    def call(self, inputs, training=True, mask=None):
        """
        Specifies how the inputs should be passed through the layers of the model and returns the output

        :param inputs: the inputs to be classified by the model, with the same shape as self.in_shape
        :param training: whether the gradients should be tracked
        :param mask: whether a certain mask should be applied on the inputs (such as masking certain timesteps)
        :return: probabilities for each of the classes
        """
        # We'll need multiple input layers and multiple convolutional layers, one for each timestep, stored in lists
        inputs_list = []

        # Create an input layer for each time step and add to the list
        for i in range(self.timeframe):
            input_layer = keras.layers.InputLayer(self.in_shape, name="input" + str(i + 1))(inputs)
            expanded = self.lamb(input_layer)
            inputs_list.append(expanded)

        # logging.INFO(len(inputs_list))
        print(inputs_list)

        # Merge the inputs together to form 'timesteps' for the RNN
        merged = self.concat(inputs_list)

        # Pass the timesteps through the RNN to find temporal features The amount of units in the layer are equal to
        # the number of timesteps (i.e. segments) according to Zhang et al. (2018)
        lstm = self.lstm(merged)

        # Final fully connected layer with softmax to give class probabilities
        output = self.out(lstm)

        return output

    def build_graph(self):
        """
        Builds the Tensor Graph in order to generate a summary of the model by running dummy input through the model
        :return: a 'dummy' version of the model of which we can generate a summary
        """
        x = keras.Input(self.in_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))


# Quick test
model = JustLSTM((248,35624),5)
zero = np.zeros((248,35624))
model.call(zero)