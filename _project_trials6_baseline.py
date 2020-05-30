import torch
import nli
import os
from sklearn.metrics import classification_report
import collections
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from torch_rnn_classifier import TorchRNNClassifier
import utils
import random
import pandas as pd

##################################################################

DATA_HOME = os.path.join("data", "nlidata")

SNLI_HOME = os.path.join(DATA_HOME, "snli_1.0")

MULTINLI_HOME = os.path.join(DATA_HOME, "multinli_1.0")

ANLI_HOME = os.path.join(DATA_HOME, "anli_v0.1")

GLOVE_HOME = os.path.join('data', 'glove.6B')

##################################################################

# Any of the files in glove.6B will work here:

glove_dim = 50

glove_src = os.path.join(GLOVE_HOME, 'glove.6B.{}d.txt'.format(glove_dim))

# Creates a dict mapping strings (words) to GloVe vectors:
GLOVE = utils.glove2dict(glove_src)

vocab = list(GLOVE.keys())
vocab.append("$UNK")

##################################################################

class RandomClassifier:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        possible_predictions = ['contradiction', 'entailment', 'neutral']
        preds = [random.choice(possible_predictions) for _ in range(len(X))]
        return preds

##################################################################

train_data = [((ex.sentence1, ex.sentence2), ex.gold_label)
       for ex in nli.MultiNLITrainReader(MULTINLI_HOME, samp_percentage=1.0).read()]

anli_map = {'c': 'contradiction', 'e': 'entailment', 'n': 'neutral'}
dev_data = [((ex.context, ex.hypothesis), anli_map[ex.label])
       for ex in nli.ANLIDevReader(ANLI_HOME, rounds=(1,)).read()]

# net = RandomClassifier()

X_glove = pd.DataFrame(GLOVE)
X_glove['$UNK'] = 0
X_glove = X_glove.T
vocab = list(X_glove.index)
embedding = X_glove.values

net = TorchRNNClassifier(
    vocab,
    embedding=embedding,
    hidden_dim=50,
    max_iter=10)

def vec_func(w):
    return w.split()

def vec_concatenate(u, v):
    """ hypothesis only baseline """
    return v

print("---------------------------------------------")

word_disjoint_experiment = nli.wordentail_experiment(
    train_data=train_data,
    assess_data=dev_data,
    model=net,
    vector_func=vec_func,
    vector_combo_func=vec_concatenate)

########################################################
# Also produce numbers for MNLI Matched/Mismatched
from nli import word_entail_featurize

def wordentail_assessonly(
        assess_data,
        vector_func,
        vector_combo_func,
        model):
    X_dev, y_dev = word_entail_featurize(
        assess_data, vector_func, vector_combo_func)
    predictions = model.predict(X_dev)
    print(classification_report(y_dev, predictions, digits=3))

dev_data = [((ex.sentence1, ex.sentence2), ex.gold_label)
       for ex in nli.MultiNLIMatchedDevReader(MULTINLI_HOME, samp_percentage=1.0).read()]

wordentail_assessonly(dev_data, vec_func, vec_concatenate, word_disjoint_experiment['model'])

dev_data = [((ex.sentence1, ex.sentence2), ex.gold_label)
       for ex in nli.MultiNLIMismatchedDevReader(MULTINLI_HOME, samp_percentage=1.0).read()]

wordentail_assessonly(dev_data, vec_func, vec_concatenate, word_disjoint_experiment['model'])
