import tensorflow as tf
import numpy as np
from tensorflow import keras


class CascadeNet(keras.Model):
    # Architecture by Zhang, D., Yao, L., Zhang, X., Wang, S., Chen, W., & Boots, R. (2018).
    # Cascade and parallel convolutional recurrent neural networks on EEG-based intention recognition for brain computer interface.
    # Proceedings of the . . . AAAI Conference on Artificial Intelligence, 32(1). https://doi.org/10.1609/aaai.v32i1.11496
    def __init__(self, input_shape, num_segments, num_classes=4):
        super().__init__()

        self.in_shape = input_shape
        self.num_classes = num_classes

        # The number of images / timesteps that we will look at for each training step
        self.num_segments = num_segments

        # naming wellicht anders doen
        self.conv1 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1")
        self.conv2 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2")
        self.conv3 = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3")

        # Flattens the 2D output of the convolutional layers to serve as input for the fully-connected layer
        self.flatten = keras.layers.Flatten(name="flatten")
        self.fc = keras.layers.Dense(8, activation="relu", name="fully_connected")

        # To get the data in the shape [batch, timesteps, features] for LSTM, we need to expand the dimensions
        # of the output of the fully connected layer that is of dimension [batch, 1024] to [batch, 1, 1024]
        self.lamb = keras.layers.Lambda(lambda previous: tf.expand_dims(previous, axis=1), name="expand_dims")

        # Concatenating the dense outputs of all timesteps (e.g. 5) gives the right input shape for the LSTM layer [batch, 5, 1024]
        self.concat = keras.layers.Concatenate(axis=1, name="concat")

        self.lstm1 = keras.layers.LSTM(self.num_segments, return_sequences=True, name="lstm1")
        self.lstm2 = keras.layers.LSTM(self.num_segments, name="lstm2")

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
        dense_convs = []

        # Create an input layer for each time step and add to the list
        for i in range(self.num_segments):
            input_layer = keras.layers.InputLayer(self.in_shape, name="input" + str(i + 1))(inputs)
            inputs_list.append(input_layer)

        # Extract spatial features for each time step separately
        for j in range(self.num_segments):
            # Padding with 0's as indicated by the paper
            conv1 = self.conv1(inputs_list[j])
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)

            flat = self.flatten(conv3)
            fc = self.fc(flat)
            expanded = self.lamb(fc)

            dense_convs.append(expanded)

        # Merge the results of the CNN together to form 'timesteps' again for the RNN part of the model
        merged = self.concat(dense_convs)

        # Pass the timesteps through the RNN to find temporal features The amount of units in the layer are equal to
        # the number of timesteps (i.e. segments) according to Zhang et al. (2018)
        lstm1 = self.lstm1(merged)
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
