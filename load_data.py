import h5py

import os
print(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name="_".join(temp)
    return dataset_name

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def read_data_file(data_path):
    with h5py.File(data_path, 'r') as f:
        dataset_name = get_dataset_name(data_path)
        matrix = f.get(dataset_name)[()]
    return matrix

def read_prepro_file(data_path):
    hfive = h5py.File(data_path, 'r')
    matrix = hfive.get('dir')
    matrix = np.array(matrix)
    return matrix

def scale(matrix, scaler, timewise=False):
    scaler_input = matrix.T if timewise else matrix
    scaled_matrix = scaler.fit_transform(scaler_input)
    return scaled_matrix.T if timewise else scaled_matrix

filename_path = "MEG_data/Final Project data/Intra/train/rest_105923_1.h5"
# OUTLIER ALERT: in "MEG_data/Final Project data/Intra/train/task_motor_105923_5.h5" konden we niet de MinMaxScaler()
# gebruiken, omdat dan meteen één outlier alles verder dichtbij 1 liet zitten


#TEST FUNCTIES
#matrix = read_data_file(filename_path)
#print(matrix.shape)

#scaled_matrix = scale(matrix, StandardScaler(), timewise=True)

#plt.imshow(scaled_matrix[:300,:1000])
#plt.colorbar()
#plt.show()

