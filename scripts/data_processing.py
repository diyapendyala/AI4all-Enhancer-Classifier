import numpy as np
import pandas as pd
from sklearn import svm, linear_model
from sklearn.model_selection import train_test_split
import sklearn.ensemble, sklearn.metrics
from torch.nn.functional import linear
import utility_functions as utils

# read in needed dataset
labeled_sequences = pd.read_csv("../data/enhancer_sequences_cleaned.tsv", sep="\t")
print(labeled_sequences)