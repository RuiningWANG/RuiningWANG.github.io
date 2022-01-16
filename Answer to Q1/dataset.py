import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18 as _resnet18
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import struct


data_train_images = np.load('data_test.npy', allow_pickle=True)
data_train_labels = np.load('labels_train.npy', allow_pickle=True)#load_mnist(path,kind = 'train')
data_test_images = np.load('data_test.npy', allow_pickle=True)
data_test_labels = np.load('labels_test.npy', allow_pickle=True)#load_mnist_test(path, kind='t10k')

class MNIST_BAG(Dataset):
    def __init__(self):
        self.data = data_train_images
        self.data = self.data.item()
        
        self.labels = data_test_labels
        self.labels = self.labels.item()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        bag, label = self.data[index], self.labels[index]
        
        return bag, label

class MNIST_BAG_VAL(Dataset):
    def __init__(self):
        self.data = data_test_images
        self.data = self.data.item()
        
        self.labels = data_test_labels
        self.labels = self.labels.item()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        bag, label = self.data[index], self.labels[index]
        
        return bag, label    

dataset = MNIST_BAG()
first_data = dataset[0]
features, labels = first_data
