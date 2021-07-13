import numpy as np
import pandas as pd
from sklearn import svm, linear_model
from sklearn.model_selection import train_test_split
import sklearn.ensemble, sklearn.metrics
from torch.nn.functional import linear
import utility_functions as utils

# read in needed datasets
file_tokens = utils.read_tokens("data/strong_weak_enhancers.txt")

# perform needed processing to assemble each of the 200 bp sequences
sequences = []
new_tuple = "FILLER" 
flag = 0
while len(file_tokens) > 0:
    token = file_tokens.pop(0)
    if flag:
        flag = False
        new_tuple = [token[1:], ""]
    elif len(file_tokens) == 0:
        new_tuple[1] += token.upper()
        sequences.append(new_tuple)
    elif token.startswith(">"):
        sequences.append(new_tuple)
        new_tuple = [token[1:], ""]
    else:
        new_tuple[1] += token.upper()

sequences_np = np.asarray(sequences[1:])
enhancer_labels = np.zeros(2968)
enhancer_labels[:1484] = 1

labeled_sequences = np.column_stack((sequences_np, enhancer_labels))
labeled_sequences_df = pd.DataFrame(labeled_sequences, columns=["name", "sequence", "enhancer_status"])

# write out cleaned data
labeled_sequences_df.to_csv("../data/enhancer_sequences_cleaned.tsv", index=False, sep="\t")

# one-hot encode the sequences
oh_sequences = np.zeros([labeled_sequences_df.shape[0], 200*4])
for i, seq in enumerate(labeled_sequences_df["sequence"]):
    oh_sequences[i] = utils.dna_onehot(seq)

x_train, x_test, y_train, y_test = train_test_split(oh_sequences, labeled_sequences_df["enhancer_status"].to_numpy(), test_size=.2, random_state=2)

# fit a random forest regression model
print("training rf")
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=2)
rf.fit(x_train, y_train)
predictions = rf.predict(x_test)

acc = sklearn.metrics.accuracy_score(y_test, predictions)
auc = sklearn.metrics.roc_auc_score(y_test, predictions)
cm = sklearn.metrics.confusion_matrix(y_test, predictions)

print("RF accuracy = ", acc, " AUC = ", auc)
print(cm)

# fit a SVM  classifier
sv = svm.SVC()
sv.fit(x_train, y_train)
predictions = sv.predict(x_test)


acc = sklearn.metrics.accuracy_score(y_test, predictions)
auc = sklearn.metrics.roc_auc_score(y_test, predictions)
cm = sklearn.metrics.confusion_matrix(y_test, predictions)

print("SVM accuracy = ", acc, " AUC = ", auc)
print(cm)

# fit a logistic regression  classifier
lr_model = linear_model.LogisticRegression(max_iter=1000)
lr_model.fit(x_train, y_train)
predictions = lr_model.predict(x_test)


acc = sklearn.metrics.accuracy_score(y_test, predictions)
auc = sklearn.metrics.roc_auc_score(y_test, predictions)
cm = sklearn.metrics.confusion_matrix(y_test, predictions)

print("Logistic regression accuracy = ", acc, " AUC = ", auc)
print(cm)


