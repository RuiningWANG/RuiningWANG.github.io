import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18 as _resnet18
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random
import numpy as np
"""
This file is run in advance and no need for running again.
The produced files are added in the directory.
"""

path ="../MNIST/"

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path+
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path+
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def load_mnist_test(path, kind='t10k'):
    """Load MNIST test data from `path`"""
    labels_path = os.path.join(path+
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path+
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

data_train = load_mnist(path)
data_test = load_mnist_test(path)
#data_train = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
#data_test = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())

indices_0 = []
indices_7 = []
indices = []
indices_test = []

for i in range(0, len(data_train)):
    img, label = data_train[i]
    
    if label == 0: 
        indices_0.append(i)

    if label == 7:
        indices_7.append(i)

    if label == 0 or label == 7:
        indices.append(i)

for i in range(0, len(data_test)):
    img, label = data_test[i]
    
    if label == 0: 
        indices_0.append(i)

    if label == 7:
        indices_7.append(i)

    if label == 0 or label == 7:
        indices_test.append(i)
    
#create 500 bags for training
train_data = {}
train_labels = {}
for i in range(0, 500):
    count_0 = 0
    count_7 = 0
    bag, label = data_train[indices[random.randint(0, 12188)]]
    
    for j in range(1, 100):
        img_temp, label_temp = data_train[indices[random.randint(0, 12187)]]
        
        bag = torch.cat((bag, img_temp), 0)
        
        if label_temp == 0:
            count_0 += 1
            
        if label_temp == 7:
            count_7 += 1
        
    train_labels[i] = torch.tensor([float(count_0/100)])
    
    train_data[i] = bag

#create 100 bags for testing
test_data = {}
test_labels = {}
for i in range(0, 100):
    count_0 = 0
    count_7 = 0
    bag, label = data_test[indices_test[random.randint(0, len(indices_test) - 1)]]
    
    for j in range(1, 100):
        img_temp, label_temp = data_train[indices[random.randint(0, len(indices_test) - 1)]]
        
        bag = torch.cat((bag, img_temp), 0)
        
        if label_temp == 0:
            count_0 += 1
            
        if label_temp == 7:
            count_7 += 1
        
    test_labels[i] = torch.tensor([float(count_0/100)])
    
    test_data[i] = bag

# np.save('data_train.npy', train_data)
# np.save('labels_train.npy', train_labels)
np.save('data_test.npy', test_data)
np.save('labels_test.npy', test_labels)







