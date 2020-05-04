from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import os
import random
import numpy as np
import matplotlib.pyplot as plt


def plot_epoch():
    losses = np.load("losses.npy", allow_pickle=True)
    epochs = list(range(1, 11))
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    ax.plot(epochs, losses[:, 0], color='g', label="Training")
    ax.plot(epochs, losses[:, 1], color='b',  label="Test")
    ax.legend()
    fig.savefig('accuraccy_by_epoch.png')
    plt.close(fig) 

def plot_subsets():
    losses = []
    subsets = np.asarray([16, 8, 4, 2])
    for i in subsets:
        temp = np.load("losses" + str(i) + ".npy", allow_pickle=True)
        losses.append(temp[-1])

    losses = np.asarray(losses)
    subsets = 60000 / subsets
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    ax.loglog(subsets, losses[:, 0], color='g', label="Training")
    ax.loglog(subsets, losses[:, 1], color='b',  label="Test")
    ax.legend()
    fig.savefig('accuraccy_by_subsets.png')
    plt.close(fig) 


plot_epoch()
plot_subsets()