import tensorflow as tf
from tensorflow import keras


# tf function zal niet heel veel uitmaken volgens de documentatie, omdat wij veel convolutional operations hebben (waarbij de speedup dus meevalt)
def train_batch(model, x_train, y_train, train_acc):
    # Instantiate an optimizer and loss function
    optimizer = keras.optimizers.Adam()
    loss_function = keras.losses.CategoricalCrossentropy()

    # Doet nu nog wel een update per window, wellicht veel? We kunnen ook nog batches gaan gebruiken om het aantal updates te reduceren
    for step in range(len(x_train)):
        # Used to track the gradients during the forward pass
        with tf.GradientTape() as tape:
            logits = model(tf.expand_dims(x_train[step], axis=0), training=True)
            # Calculate the loss value
            loss_value = loss_function(y_train[step], logits)

        # Extract the gradients from the gradienttape
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Perform a weight update step using the extracted gradients
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # Keep track of the training accuracy which is given at the end of the loop
        train_acc.update_state(y_train[step], logits)

        # Report the training loss for monitoring
        if step % 100 == 0:
            print("Training loss value at step {}: {}".format(step, loss_value))

    print("Training accuracy over whole file: {}".format(train_acc.result()))
