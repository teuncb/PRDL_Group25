from lstm import JustLSTM
from sklearn.preprocessing import LabelBinarizer
import training
import tensorflow
from tensorflow import keras

# Hyperparameters
timeframe = 5
lstm_units = 5
learning_rate = 1e-2
output_classes = 4



textual_labels = ["rest", "task_motor", "task_story_math", "task_working_memory"]
label_encoder = LabelBinarizer()
label_encoder.fit(textual_labels)

model = JustLSTM((timeframe, 248), timeframe, lstm_units, 16, output_classes)
model_type = "lstm"

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

training.train_dir(model, model_type, "Final Project data/Cross/train_prepro", 1, timeframe, optimizer, label_encoder)