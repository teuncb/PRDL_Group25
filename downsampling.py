import numpy as np
from skimage.measure import block_reduce

def downsample_matrix(matrix, n, leave_out=True):
    """_summary_

    Args:
        matrix (ndarray): The matrix we want to downsample.
        n (int): The number of columns we want to leave out. I.e. if n=5,
            4 columns will be skipped.
        leave_out (bool, optional): Whether we use drop, we average over the
            skipped columns if False. Defaults to True.

    Returns:
        matrix (ndarray): The downsampled matrix.
    """
    if leave_out:
        # Leave out each non-nth column.
        return matrix[:, ::n]

    means = block_reduce(
        matrix,
        block_size=(1, n),
        func=np.mean,
        cval=np.mean(matrix),
    )

    return means
