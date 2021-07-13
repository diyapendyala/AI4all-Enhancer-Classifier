### Import all the necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import utility_functions as utils
from nn_classes import MyNet, SampleCNN
from sklearn.model_selection import train_test_split
import sklearn.ensemble, sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

# read in dataset
needed_data = pd.read_csv("../data/original_cleaned_data_pgl4.tsv", sep="\t")
# remove some unneeded rows
needed_data = needed_data.dropna(subset=["pGL4 mean log2 RNA/DNA ratio"])
needed_data = needed_data[needed_data.type.isin(["test"])]

# one-hot encode the sequences
oh_sequences = np.zeros([needed_data.shape[0], 171*4])
for i, seq in enumerate(needed_data["sequence (171nt)"]):
    oh_sequences[i] = utils.dna_onehot(seq)

x_train, x_test, y_train, y_test = train_test_split(oh_sequences, needed_data["pGL4 mean log2 RNA/DNA ratio"].to_numpy(), test_size=.2, random_state=1)

x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)
x_test, y_test = torch.Tensor(x_test), torch.Tensor(y_test)

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

# put the data into dataloaders 
BATCH_SIZE = 128
NUM_WORKERS = 0

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=NUM_WORKERS)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=NUM_WORKERS)

# make a model, optimizer, and loss
fake_model = MyNet()
adam_optimizer = torch.optim.Adam(fake_model.parameters(), lr=.001)
loss_criterion = torch.nn.MSELoss()

# train a fully connected NN
losses = utils.train(train_loader, fake_model, adam_optimizer, loss_criterion, n_epochs=100)
utils.test(fake_model, loss_criterion, test_loader)

############################################### 
############# now lets try a CNN ###############
###############################################

# make a model, optimizer, and loss
fake_model = SampleCNN()
adam_optimizer = torch.optim.Adam(fake_model.parameters(), lr=.001)
loss_criterion = torch.nn.MSELoss()

# train a fully connected NN
losses = utils.train(train_loader, fake_model, adam_optimizer, loss_criterion, n_epochs=100)
utils.test(fake_model, loss_criterion, test_loader)
