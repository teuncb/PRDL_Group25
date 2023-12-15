import h5py

def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name="_".join(temp)
    return dataset_name

import os
print(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
def sigmoid(x):
    return 1/(1 + np.exp(-x))

filename_path = "MEG_data/Final Project data/Intra/train/task_motor_105923_5.h5"
# OUTLIER ALERT: in "MEG_data/Final Project data/Intra/train/task_motor_105923_5.h5" konden we niet de MinMaxScaler()
# gebruiken, omdat dan meteen één outlier alles verder dichtbij 1 liet zitten

with h5py.File(filename_path,'r') as f:
    dataset_name=get_dataset_name(filename_path)
    matrix=f.get(dataset_name)[()]
    print(type(matrix))
    print(matrix.shape)

    print(matrix[:20])

    mmscaler = MinMaxScaler()
    mmscaler.fit(matrix[:300,:1000])
    larger = mmscaler.transform(matrix[:300,:1000])
    print(larger)

    plt.imshow(larger)
    plt.colorbar()
    plt.show()
