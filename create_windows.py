import numpy as np
from data_loading import array_to_mesh, create_meshes
import matplotlib.pyplot as plt
import load_data


def create_windows(dataset, timeframe):
    # Will contain the slices of the dataset that represent the different windows
    windows = []

    num_windows = int(dataset.shape[1] / timeframe)

    i = timeframe
    j = 0
    count = 0

    # Loop through the dataset until we have the specified number of windows (might cut off some of the last timesteps)
    while count < num_windows:
        # Save the view of the array in the windows list
        view = dataset[:, j:i]
        windows.append(view)

        i += timeframe
        j += timeframe
        count += 1

    # Return the list of array views
    return windows


# Lil bit of testing, alle dingen uit deze file gaan later wss naar andere
test_prepro = load_data.read_prepro_file("Final Project data/Intra/train_prepro/rest_105923_1.h5")
windows = create_windows(test_prepro, 5)
print(len(windows))

meshes = create_meshes(windows[0])
for i in range(meshes.shape[2]):
    plt.imshow(meshes[:, :, i])
    plt.colorbar()
    plt.show()
print(meshes.shape)

# Gepreprocessde dataset heeft shape: (248, 11.875)
