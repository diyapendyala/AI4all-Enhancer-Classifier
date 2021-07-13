import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn.metrics
from sklearn.utils import validation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms

def read_tokens(filename):
    infile = open(filename, "r")
    contents = infile.read()
    infile.close()

    split_contents = contents.split()
    return split_contents

def dna_onehot(seq):
    oh_vector = np.zeros(len(seq)*4)
    for i, base in enumerate(seq):
        if base == "A":
            oh_vector[i*4] = 1
        elif base == "C":
            oh_vector[(i*4) + 1] = 1
        elif base == "T":
            oh_vector[(i*4) + 2] = 1
        elif base == "G":
            oh_vector[(i*4) + 3] = 1
        else:
            raise ValueError("DNA base " + base + " not recognized")
    
    return oh_vector

def dna_onehot2d(seq):
    oh_vector = []
    for base in seq:
        if base == "A":
            oh_vector.append([1, 0, 0, 0])
        elif base == "C":
            oh_vector.append([0, 1, 0, 0])
        elif base == "T":
            oh_vector.append([0, 0, 1, 0])
        elif base == "G":
            oh_vector.append([0, 0, 0, 1])
        else:
            raise ValueError("DNA base " + base + " not recognized")
    
    return np.asarray(oh_vector)


def train(train_loader, model, optimizer, criterion, 
          n_epochs=10, savename="", save_intermediates = False, validation_loader=None):
    
    
    losses = []

    for epoch in range(n_epochs):
        epoch_losses = 0
        model.train()  # set the model's parameters to be trainable
        for data, labels in train_loader:
            optimizer.zero_grad()

            output = model.forward(data)
            output = torch.squeeze(output)
            loss = criterion(output, labels)
            # bookkeeping
            epoch_losses += loss.item()

            # perform backprop
            loss.backward()
            optimizer.step()
            
            # bookkeeping
            epoch_losses += loss.item()
        
        # save model every 5 epochs, if needed
        if ((epoch+1) % 5) == 0 and save_intermediates:
            pathname = "model_" + savename + str(epoch+1) + ".txt"
            torch.save(model.state_dict(), pathname)

        # compute accuracy and average loss
        epoch_loss = epoch_losses / len(train_loader.dataset)
        print("Epoch", epoch, "average training loss:", epoch_loss)
        losses.append(epoch_loss)
        
        if validation_loader is not None:
            test_loss = 0
            predictions = []
            truth = []

            with torch.no_grad():
                model.eval()
                for data, labels in validation_loader:

                    output = model.forward(data)
                    output = torch.squeeze(output)
            
                    test_loss += criterion(output, labels).item()
                    for y_hat, y in zip(output, labels):
                        predictions.append(y_hat.item())
                        truth.append(y.item())

            #predictions = [sum(truth)/len(truth) for i in range(len(predictions))]
            test_loss /= len(validation_loader.dataset)
            r2 = sklearn.metrics.r2_score(truth, predictions)
            print("Validation loss: ", test_loss, "\nValidation R2: ",r2)

    return losses



def test(model, 
         criterion,
         test_loader):
    
    test_loss = 0
    predictions = []
    truth = []

    with torch.no_grad():
        model.eval()
        for data, labels in test_loader:

            output = model.forward(data)
            output = torch.squeeze(output)
    
            test_loss += criterion(output, labels).item()
            for y_hat, y in zip(output, labels):
                predictions.append(y_hat.item())
                truth.append(y.item())

    #predictions = [sum(truth)/len(truth) for i in range(len(predictions))]
    test_loss /= len(test_loader.dataset)
    r2 = sklearn.metrics.r2_score(truth, predictions)
    print("Test loss: ", test_loss, "\nTest R2: ",r2)

