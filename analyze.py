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

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.conv3 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.1)
        self.fc1 = nn.Linear(968, 32)
        self.fc2 = nn.Linear(32, 10)
        self.bn1 = nn.BatchNorm2d(8)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output, x

def plot_epoch():
    losses = np.load("losses.npy", allow_pickle=True)
    epochs = list(range(1, 11))
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    ax.plot(epochs, losses[:, 0], color='g', label="Training Loss")
    ax.plot(epochs, losses[:, 1], color='b',  label="Test Loss")
    ax.legend()
    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Loss")
    fig.savefig('accuracy_by_epoch.png')
    plt.close(fig) 

def plot_subsets():
    losses = []
    subsets = np.asarray([16, 8, 4, 2])
    for i in subsets:
        temp = np.load("losses" + str(i) + ".npy", allow_pickle=True)
        print(temp[-1])
        losses.append(temp[-1])

    losses = np.asarray(losses)
    subsets = 60000 / subsets
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    ax.loglog(subsets, losses[:, 0], color='g', label="Training")
    ax.loglog(subsets, losses[:, 1], color='b',  label="Test")
    ax.legend()
    ax.set_xlabel("Number of Training Examples")
    ax.set_ylabel("Loss")
    fig.savefig('accuraccy_by_subsets.png')
    plt.close(fig) 

def mistaken_classifier(out):
    fig, ax = plt.subplots( nrows=3, ncols=3 ) 
    for i in range(3):
        for j in range(3):
            ax[i][j].imshow(np.squeeze(np.asarray(out[i*3 + j][0])), cmap="gray")
            ax[i][j].set_title("Pred: " + str(out[i*3 + j][1].item()) + " Target: " + str(out[i*3 + j][2].item()))
    plt.savefig("mistaken_classifier.png")
    plt.close(fig) 


def learned_kernels(model):
    weights = model.conv1.weight.data
    fig, ax = plt.subplots( nrows=3, ncols=3 ) 
    for i in range(3):
        for j in range(3):
            if i*3 + j >= 8:
                i = 3
                j = 3
                break
            ax[i][j].imshow(np.squeeze(np.asarray(weights[i*3 + j])), cmap="gray")
    fig.savefig("learned_kernels.png")
    plt.close(fig) 

def gen_confusion_matrix(y_target, y_pred):
    mat = confusion_matrix(y_target, y_pred)
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    im = ax.imshow(mat, origin='lower', interpolation='None', cmap='viridis')
    for i in range(10):
        for j in range(10):
            label = mat[i, j]
            ax.text(i, j, label, color='black', ha='center', va='center')

    fig.colorbar(im)
    fig.savefig('confusion_matrix.png')
    plt.close(fig) 
                

def visual_tsne(y_target, emb):
    colormap = plt.cm.gist_ncar
    colors = [colormap(i) for i in np.linspace(0, 0.9, 10)]  

    t = TSNE(n_components=2).fit_transform(emb)
    labels = [[], [], [], [], [], [], [], [], [], []]
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 

    for i in range(len(y_target)):
        labels[y_target[i]].append(t[i,:])
    for i in range(10):
        ax.scatter(np.asarray(labels[i])[:, 0], np.asarray(labels[i])[:, 1], color=colors[i], label=i)
    
    ax.legend()
    fig.savefig('tsne.png')
    plt.close(fig)


def feature_vector(data, emb):
    fig, ax = plt.subplots(4, 9)
    for i in range(4):
        e = emb[i]
        euclid = np.linalg.norm(emb - e, axis=1)
        nghbr = euclid[np.argpartition(euclid, 8)[:8]]
        idx = []
        for n in nghbr:
            idx.append(np.argwhere(euclid == n)[0])

        ax[i][0].imshow(data[i], cmap='gray')
        ax[i][0].set_title('I_0')

        for k in range(8):
            ax[i][k+1].imshow(data[idx[k][0]], cmap='gray')
            ax[i][k+1].set_title('I_' + str(k + 1))
    
    fig.savefig('feature_vector.png')
    return


def main():
    plot_epoch()
    plot_subsets()

    device = torch.device("cpu")
    torch.manual_seed(1)
    kwargs = {}
    model = Net().to(device)
    model.load_state_dict(torch.load("mnist_model.pt"))

    test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1000, shuffle=True, **kwargs)
    
    y_pred = []
    y_target = []
    out = []
    emb = []
    data_list = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, e = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            for i in range(len(data)):
                y_pred.append(pred[i])
                y_target.append(target[i])
                emb.append(e[i].tolist())
                data_list.append(data[i, 0])
                if pred[i] != target[i]:
                    out.append((data[i], pred[i], target[i]))

    mistaken_classifier(out)
    learned_kernels(model)
    gen_confusion_matrix(y_target, y_pred)
    visual_tsne(y_target, emb)
    feature_vector(data_list, np.asarray(emb))



if __name__ == '__main__':
    main()

