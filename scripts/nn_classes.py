### Import all the necessary libraries
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import MaxPool1d
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(684, 2000) # MNIST images are 28 x 28
        self.fc1 = nn.Linear(2000, 1) # MNIST images are 28 x 28
        self.fc2 = nn.Linear(1000, 500) # from 200-node layer to 100 node layer
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 1) # from 100-node layer to output layer

    
    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))  # Pass the input through fc1 and apply relu nonlinearity
        #x = F.relu(self.fc2(x))  # Pass the output from the 1st layer into the 2nd layer and apply relu
        #x = F.relu(self.fc3(x))
        #x = self.fc4(x) # Pass from the second layer to the output layer
        return x # returns the output of the final layer

class SampleCNN(nn.Module):
    def __init__(self):
        super(SampleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=50, kernel_size=4, padding=2),
            nn.Dropout(.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=100, kernel_size=4, padding=2),
            nn.Dropout(.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=500, kernel_size=4, padding=2),
            nn.Dropout(.2)
        )
        
        self.fc1 = nn.Linear(685*50, 250)
        self.fc2 = nn.Linear(250, 1)
    
    def forward(self, x):
        # input x : BATCH_SIZE x 64 x 1
        # expected conv1d input : minibatch_size x num_channel x width
        x = x.view(x.shape[0], 1,-1)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        #x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x