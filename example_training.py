from sklearn.preprocessing import LabelBinarizer
import tensorflow
from tensorflow import keras
from training import train_dir
from eval import evaluate_model

from sklearn.preprocessing import LabelBinarizer


dirpath = "/content/content/MyDrive/train_prepro_intra"
test_paths = ["/content/content/MyDrive/val_prepro"]

output_classes = 4

# Hyperparameters
learning_rates = [1e-2, 1e-3, 1e-4]
timeframes = [5, 10, 50]
LSTM_units = [20, 50]

# Labelencoder
textual_labels = ["rest", "task_motor", "task_story_math", "task_working_memory"]
label_encoder = LabelBinarizer()
label_encoder.fit(textual_labels)

accuracies = []
average_performance_metrics = []

for learning_rate in learning_rates:
  for timeframe in timeframes:
    for LSTM_unit in LSTM_units:
      print("Training model with lr={}, timeframe={}, units={}".format(learning_rate, timeframe, LSTM_unit))
      print()

      model = Single_Layer_LSTM((timeframe, 248), timeframe, LSTM_unit)

      model.compile(loss=keras.losses.CategoricalCrossentropy(),
                    metrics=keras.metrics.CategoricalAccuracy(),
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate))



      mod, history = train_dir(model, "lstm", dirpath , 20, timeframe, label_encoder)
      accuracy, average_performance_metric = evaluate_model("lstm", model, test_paths, timeframe)