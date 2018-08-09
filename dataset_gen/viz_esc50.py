import matplotlib.pyplot as plt
import numpy as np
import os


DATASET_PATH = '/home/moreaux-gpu/Dataset/ENVNET_DB/esc10/wav16.npz'

dataset = np.load(DATASET_PATH)
sounds = dataset['fold1'].item()['sounds']
lbls = dataset['fold1'].item()['labels']

n_items = 5
fig, axs = plt.subplots(n_items, n_items)
for i in range(n_items * n_items):
    ax = axs[i / n_items][i % n_items]
    sound = sounds[i] / (25000.)
    ax.plot(sound)
    ax.set_ylim(-1, 1)
    ax.set_title(lbls[i])
    ax.set_axis_off()
fig.show()
