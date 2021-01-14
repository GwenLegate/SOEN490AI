import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import constants as c

def show_activations(file, batch_no, row, col):
    activations = np.load(file, allow_pickle=True)
    maps, x, y = activations[batch_no, :, :, :].shape

    plt.figure(figsize=(row*2, col*2))
    for m in range(maps):
        plt.subplot(row, col, m+1)
        plt.imshow(activations[batch_no, m, :, :], cmap='summer', interpolation='nearest')
    plt.show()

show_activations(c.ACTIVATION_LYR_1, 0, 4, 2)
