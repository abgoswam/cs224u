#!/usr/bin/env python
# coding: utf-8

# # Homework and bake-off: word-level entailment with neural networks

# In[1]:


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


# ## Contents
# 
# 1. [Overview](#Overview)
# 1. [Set-up](#Set-up)
# 1. [Data](#Data)
#   1. [Edge disjoint](#Edge-disjoint)
#   1. [Word disjoint](#Word-disjoint)
# 1. [Baseline](#Baseline)
#   1. [Representing words: vector_func](#Representing-words:-vector_func)
#   1. [Combining words into inputs: vector_combo_func](#Combining-words-into-inputs:-vector_combo_func)
#   1. [Classifier model](#Classifier-model)
#   1. [Baseline results](#Baseline-results)
# 1. [Homework questions](#Homework-questions)
#   1. [Hypothesis-only baseline [2 points]](#Hypothesis-only-baseline-[2-points])
#   1. [Alternatives to concatenation [2 points]](#Alternatives-to-concatenation-[2-points])
#   1. [A deeper network [2 points]](#A-deeper-network-[2-points])
#   1. [Your original system [3 points]](#Your-original-system-[3-points])
# 1. [Bake-off [1 point]](#Bake-off-[1-point])

# ## Overview

# The general problem is word-level natural language inference.
# 
# Training examples are pairs of words $(w_{L}, w_{R}), y$ with $y = 1$ if $w_{L}$ entails $w_{R}$, otherwise $0$.
# 
# The homework questions below ask you to define baseline models for this and develop your own system for entry in the bake-off, which will take place on a held-out test-set distributed at the start of the bake-off. (Thus, all the data you have available for development is available for training your final system before the bake-off begins.)
# 
# <img src="fig/wordentail-diagram.png" width=600 alt="wordentail-diagram.png" />

# ## Set-up

# See [the first notebook in this unit](nli_01_task_and_data.ipynb) for set-up instructions.

# In[2]:


from collections import defaultdict
import json
import numpy as np
import os
import pandas as pd
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
import nli
import utils


# In[3]:


DATA_HOME = 'data'

NLIDATA_HOME = os.path.join(DATA_HOME, 'nlidata')

wordentail_filename = os.path.join(
    NLIDATA_HOME, 'nli_wordentail_bakeoff_data.json')

GLOVE_HOME = os.path.join(DATA_HOME, 'glove.6B')


# ## Data
# 
# I've processed the data into two different train/test splits, in an effort to put some pressure on our models to actually learn these semantic relations, as opposed to exploiting regularities in the sample.
# 
# * `edge_disjoint`: The `train` and `dev` __edge__ sets are disjoint, but many __words__ appear in both `train` and `dev`.
# * `word_disjoint`: The `train` and `dev` __vocabularies are disjoint__, and thus the edges are disjoint as well.
# 
# These are very different problems. For `word_disjoint`, there is real pressure on the model to learn abstract relationships, as opposed to memorizing properties of individual words.

# In[4]:


with open(wordentail_filename) as f:
    wordentail_data = json.load(f)


# The outer keys are the  splits plus a list giving the vocabulary for the entire dataset:

# In[5]:


wordentail_data.keys()


# ### Edge disjoint

# In[6]:


wordentail_data['edge_disjoint'].keys()


# This is what the split looks like; all three have this same format:

# In[7]:


wordentail_data['edge_disjoint']['dev'][: 5]


# Let's test to make sure no edges are shared between `train` and `dev`:

# In[8]:


nli.get_edge_overlap_size(wordentail_data, 'edge_disjoint')


# As we expect, a *lot* of vocabulary items are shared between `train` and `dev`:

# In[9]:


nli.get_vocab_overlap_size(wordentail_data, 'edge_disjoint')


# This is a large percentage of the entire vocab:

# In[10]:


len(wordentail_data['vocab'])


# Here's the distribution of labels in the `train` set. It's highly imbalanced, which will pose a challenge for learning. (I'll go ahead and reveal that the `dev` set is similarly distributed.)

# In[11]:


def label_distribution(split):
    return pd.DataFrame(wordentail_data[split]['train'])[1].value_counts()


# In[12]:


label_distribution('edge_disjoint')


# ### Word disjoint

# In[13]:


wordentail_data['word_disjoint'].keys()


# In the `word_disjoint` split, no __words__ are shared between `train` and `dev`:

# In[14]:


nli.get_vocab_overlap_size(wordentail_data, 'word_disjoint')


# Because no words are shared between `train` and `dev`, no edges are either:

# In[15]:


nli.get_edge_overlap_size(wordentail_data, 'word_disjoint')


# The label distribution is similar to that of `edge_disjoint`, though the overall number of examples is a bit smaller:

# In[16]:


label_distribution('word_disjoint')


# ## Baseline

# Even in deep learning, __feature representation is vital and requires care!__ For our task, feature representation has two parts: representing the individual words and combining those representations into a single network input.

# ### Representing words: vector_func

# Let's consider two baseline word representations methods:
# 
# 1. Random vectors (as returned by `utils.randvec`).
# 1. 50-dimensional GloVe representations.

# In[17]:


def randvec(w, n=50, lower=-1.0, upper=1.0):
    """Returns a random vector of length `n`. `w` is ignored."""
    return utils.randvec(n=n, lower=lower, upper=upper)


# In[18]:


# Any of the files in glove.6B will work here:

glove_dim = 50

glove_src = os.path.join(GLOVE_HOME, 'glove.6B.{}d.txt'.format(glove_dim))

# Creates a dict mapping strings (words) to GloVe vectors:
GLOVE = utils.glove2dict(glove_src)

def glove_vec(w):    
    """Return `w`'s GloVe representation if available, else return 
    a random vector."""
    return GLOVE.get(w, randvec(w, n=glove_dim))


# ### Combining words into inputs: vector_combo_func

# Here we decide how to combine the two word vectors into a single representation. In more detail, where `u` is a vector representation of the left word and `v` is a vector representation of the right word, we need a function `vector_combo_func` such that `vector_combo_func(u, v)` returns a new input vector `z` of dimension `m`. A simple example is concatenation:

# In[19]:


def vec_concatenate(u, v):
    """Concatenate np.array instances `u` and `v` into a new np.array"""
    return np.concatenate((u, v))


# `vector_combo_func` could instead be vector average, vector difference, etc. (even combinations of those) – there's lots of space for experimentation here; [homework question 2](#Alternatives-to-concatenation-[1-point]) below pushes you to do some exploration.

# ### Classifier model
# 
# For a baseline model, I chose `TorchShallowNeuralClassifier`:

# In[20]:


net = TorchShallowNeuralClassifier(hidden_dim=50, max_iter=100)


# ### Baseline results
# 
# The following puts the above pieces together, using `vector_func=glove_vec`, since `vector_func=randvec` seems so hopelessly misguided for `word_disjoint`!

# In[21]:


word_disjoint_experiment = nli.wordentail_experiment(
    train_data=wordentail_data['word_disjoint']['train'],
    assess_data=wordentail_data['word_disjoint']['dev'], 
    model=net, 
    vector_func=glove_vec,
    vector_combo_func=vec_concatenate)

print("macro-f1: {0}".format(word_disjoint_experiment['macro-F1']))


# ## Homework questions
# 
# Please embed your homework responses in this notebook, and do not delete any cells from the notebook. (You are free to add as many cells as you like as part of your responses.)

# ### Hypothesis-only baseline [2 points]
# 
# During our discussion of SNLI and MultiNLI, we noted that a number of research teams have shown that hypothesis-only baselines for NLI tasks can be remarkably robust. This question asks you to explore briefly how this baseline effects the 'edge_disjoint' and 'word_disjoint' versions of our task.
# 
# For this problem, submit two functions:
# 
# 1. A `vector_combo_func` function called `hypothesis_only` that simply throws away the premise, using the unmodified hypothesis (second) vector as its representation of the example.
# 
# 1. A function called `run_hypothesis_only_evaluation` that does the following:
#     1. Loops over the two conditions 'word_disjoint' and 'edge_disjoint' and the two `vector_combo_func` values `vec_concatenate` and `hypothesis_only`, calling `nli.wordentail_experiment` to train on the conditions 'train' portion and assess on its 'dev' portion, with `glove_vec` as the `vector_func`. So that the results are consistent, use an `sklearn.linear_model.LogisticRegression` with default parameters as the model.
#     1. Returns a `dict` mapping `(condition_name, function_name)` pairs to the 'macro-F1' score for that pair, as returned by the call to `nli.wordentail_experiment`. (Tip: you can get the `str` name of your function `hypothesis_only` with `hypothesis_only.__name__`.)
#     
# The test functions `test_hypothesis_only` and `test_run_hypothesis_only_evaluation` will help ensure that your functions have the desired logic.

# In[22]:


##### YOUR CODE HERE
def hypothesis_only(u, v):
    """Just return the hypothesis part"""
    return v


def run_hypothesis_only_evaluation():
    ##### YOUR CODE HERE
    
    from sklearn.linear_model import LogisticRegression

    eval_results = {}
    net = LogisticRegression()
    for condition_name in ['edge_disjoint', 'word_disjoint']:
        for vec_combo_func in [vec_concatenate, hypothesis_only]:
            result = nli.wordentail_experiment(
                train_data=wordentail_data[condition_name]['train'],
                assess_data=wordentail_data[condition_name]['dev'],
                model=net,
                vector_func=glove_vec,
                vector_combo_func=vec_combo_func)

            print("macro-f1: {0}".format(result['macro-F1']))
            eval_results[(condition_name, vec_combo_func.__name__)] = result['macro-F1']

    return eval_results


# In[23]:


def test_hypothesis_only(hypothesis_only):
    v = hypothesis_only(1, 2)
    assert v == 2   


# In[24]:


test_hypothesis_only(hypothesis_only)


# In[25]:


def test_run_hypothesis_only_evaluation(run_hypothesis_only_evaluation):
    results = run_hypothesis_only_evaluation()
    assert ('word_disjoint', 'vec_concatenate') in results,         "The return value of `run_hypothesis_only_evaluation` does not have the intended kind of keys"
    assert isinstance(results[('word_disjoint', 'vec_concatenate')], float),         "The values of the `run_hypothesis_only_evaluation` result should be floats"


# In[26]:


test_run_hypothesis_only_evaluation(run_hypothesis_only_evaluation)


# ### Alternatives to concatenation [2 points]
# 
# We've so far just used vector concatenation to represent the premise and hypothesis words. This question asks you to explore two simple alternative:
# 
# 1. Write a function `vec_diff` that, for a given pair of vector inputs `u` and `v`, returns the element-wise difference between `u` and `v`.
# 
# 1. Write a function `vec_max` that, for a given pair of vector inputs `u` and `v`, returns the element-wise max values between `u` and `v`.
# 
# You needn't include your uses of `nli.wordentail_experiment` with these functions, but we assume you'll be curious to see how they do!

# In[27]:


def vec_diff(u, v):
    ##### YOUR CODE HERE
    return u - v



    
def vec_max(u, v):
    ##### YOUR CODE HERE
    return np.maximum(u, v)



# In[28]:


def test_vec_diff(vec_diff):
    u = np.array([10.2, 8.1])
    v = np.array([1.2, -7.1])
    result = vec_diff(u, v)
    expected = np.array([9.0, 15.2])
    assert np.array_equal(result, expected),         "Expected {}; got {}".format(expected, result)


# In[29]:


test_vec_diff(vec_diff)


# In[30]:


def test_vec_max(vec_max):
    u = np.array([1.2,  8.1])
    v = np.array([10.2, -7.1])
    result = vec_max(u, v)
    expected = np.array([10.2, 8.1])
    assert np.array_equal(result, expected),         "Expected {}; got {}".format(expected, result)


# In[31]:


test_vec_max(vec_max)


# In[32]:


if 'IS_GRADESCOPE_ENV' not in os.environ:
    
    net = TorchShallowNeuralClassifier(hidden_dim=50, max_iter=100)
    print(net)
    
    for vec_combo_func in [vec_diff, vec_max, vec_concatenate, hypothesis_only]:
        result = nli.wordentail_experiment(
            train_data=wordentail_data['word_disjoint']['train'],
            assess_data=wordentail_data['word_disjoint']['dev'],
            model=net,
            vector_func=glove_vec,
            vector_combo_func=vec_combo_func)

        print("macro-f1: {0}".format(result['macro-F1']))


# ### A deeper network [2 points]
# 
# It is very easy to subclass `TorchShallowNeuralClassifier` if all you want to do is change the network graph: all you have to do is write a new `define_graph`. If your graph has new arguments that the user might want to set, then you should also redefine `__init__` so that these values are accepted and set as attributes.
# 
# For this question, please subclass `TorchShallowNeuralClassifier` so that it defines the following graph:
# 
# $$\begin{align}
# h_{1} &= xW_{1} + b_{1} \\
# r_{1} &= \textbf{Bernoulli}(1 - \textbf{dropout\_prob}, n) \\
# d_{1} &= r_1 * h_{1} \\
# h_{2} &= f(d_{1}) \\
# h_{3} &= h_{2}W_{2} + b_{2}
# \end{align}$$
# 
# Here, $r_{1}$ and $d_{1}$ define a dropout layer: $r_{1}$ is a random binary vector of dimension $n$, where the probability of a value being $1$ is given by $1 - \textbf{dropout_prob}$. $r_{1}$ is multiplied element-wise by our first hidden representation, thereby zeroing out some of the values. The result is fed to the user's activation function $f$, and the result of that is fed through another linear layer to produce $h_{3}$. (Inside `TorchShallowNeuralClassifier`, $h_{3}$ is the basis for a softmax classifier, so no activation function is applied to it.)
# 
# For your implementation, please use `nn.Sequential`, `nn.Linear`, and `nn.Dropout` to define the required layers.
# 
# For comparison, using this notation, `TorchShallowNeuralClassifier` defines the following graph:
# 
# $$\begin{align}
# h_{1} &= xW_{1} + b_{1} \\
# h_{2} &= f(h_{1}) \\
# h_{3} &= h_{2}W_{2} + b_{2}
# \end{align}$$
# 
# The following code starts this sub-class for you, so that you can concentrate on `define_graph`. Be sure to make use of `self.dropout_prob`
# 
# For this problem, submit just your completed  `TorchDeepNeuralClassifier`. You needn't evaluate it, though we assume you will be keen to do that!
# 
# You can use `test_TorchDeepNeuralClassifier` to ensure that your network has the intended structure.

# In[33]:


import torch.nn as nn

class TorchDeepNeuralClassifier(TorchShallowNeuralClassifier):
    def __init__(self, dropout_prob=0.7, **kwargs):
        self.dropout_prob = dropout_prob
        super().__init__(**kwargs)
    
    def define_graph(self):
        """Complete this method!
        
        Returns
        -------
        an `nn.Module` instance, which can be a free-standing class you 
        write yourself, as in `torch_rnn_classifier`, or the outpiut of 
        `nn.Sequential`, as in `torch_shallow_neural_classifier`.
        
        """
        ##### YOUR CODE HERE
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Dropout(p=self.dropout_prob),
            self.hidden_activation,
            nn.Linear(self.hidden_dim, self.n_classes_))


    

##### YOUR CODE HERE  
if 'IS_GRADESCOPE_ENV' not in os.environ:
    net = TorchDeepNeuralClassifier()
    for vec_combo_func in [vec_diff, vec_max, vec_concatenate, hypothesis_only]:
        result = nli.wordentail_experiment(
            train_data=wordentail_data['word_disjoint']['train'],
            assess_data=wordentail_data['word_disjoint']['dev'],
            model=net,
            vector_func=glove_vec,
            vector_combo_func=vec_combo_func)

        print("macro-f1: {0}".format(result['macro-F1']))


# In[34]:


def test_TorchDeepNeuralClassifier(TorchDeepNeuralClassifier):
    dropout_prob = 0.55
    assert hasattr(TorchDeepNeuralClassifier(), "dropout_prob"),         "TorchDeepNeuralClassifier must have an attribute `dropout_prob`."
    try:
        inst = TorchDeepNeuralClassifier(dropout_prob=dropout_prob)
    except TypeError:
        raise TypeError("TorchDeepNeuralClassifier must allow the user "
                        "to set `dropout_prob` on initialization")
    inst.input_dim = 10
    inst.n_classes_ = 5
    graph = inst.define_graph()
    assert len(graph) == 4,         "The graph should have 4 layers; yours has {}".format(len(graph))    
    expected = {
        0: 'Linear',
        1: 'Dropout',
        2: 'Tanh',
        3: 'Linear'}
    for i, label in expected.items():
        name = graph[i].__class__.__name__
        assert label in name,             "The {} layer of the graph should be a {} layer; yours is {}".format(i, label, name)
    assert graph[1].p == dropout_prob,         "The user's value for `dropout_prob` should be the value of `p` for the Dropout layer."


# In[35]:


test_TorchDeepNeuralClassifier(TorchDeepNeuralClassifier)


# ### Your original system [3 points]
# 
# This is a simple dataset, but our focus on the 'word_disjoint' condition ensures that it's a challenging one, and there are lots of modeling strategies one might adopt. 
# 
# You are free to do whatever you like. We require only that your system differ in some way from those defined in the preceding questions. They don't have to be completely different, though. For example, you might want to stick with the model but represent examples differently, or the reverse.
# 
# Keep in mind that, for the bake-off evaluation, the 'edge_disjoint' portions of the data are off limits. You can, though, train on the combination of the 'word_disjoint' 'train' and 'dev' portions. You are free to use different pretrained word vectors and the like. Please do not introduce additional entailment datasets into your training data, though.
# 
# Please embed your code in this notebook so that we can rerun it.
# 
# In the cell below, please provide a brief technical description of your original system, so that the teaching team can gain an understanding of what it does. This will help us to understand your code and analyze all the submissions to identify patterns and strategies.

# In[40]:


# Enter your system description in this cell.

# Tried out different systems:
# system_0 : Original system. 
#            Uses bidirectional RNN classifier (0.68)
#            Choosing this as original system.

# system_1 : Variations of vector_combo_func (0.70)
# system_2 : Retrofit GLOVE using WordNet (0.64)
# system_3 : Data augmentation of 'entails' class using WordNet to avoid imbalance during training. (0.48)

# My peak score was: 0.68

if 'IS_GRADESCOPE_ENV' not in os.environ:
    from nltk.corpus import wordnet as wn
    from retrofitting import Retrofitter
    from torch_rnn_classifier import TorchRNNClassifier
    
    def get_wordnet_edges():
        edges = defaultdict(set)
        for ss in wn.all_synsets():
            lem_names = {lem.name() for lem in ss.lemmas()}
            for lem in lem_names:
                edges[lem] |= lem_names
        return edges

    wn_edges = get_wordnet_edges()
    
    # Idea: Bidirectional RNN classifier
    def system_0_original():
        
        # Data------------     
        with open(wordentail_filename) as f:
            wordentail_data = json.load(f)
        
        print("Distribution of labels : \n{0}".format(pd.DataFrame(wordentail_data['word_disjoint']['train'])[1].value_counts()))

        # Model-----------
        X_glove = pd.DataFrame(GLOVE)
        X_glove['$UNK'] = 0
        X_glove = X_glove.T
    
        vocab = list(X_glove.index)
        embedding = X_glove.values
        net = TorchRNNClassifier(vocab=vocab, embedding=embedding, bidirectional=True)
        
        # Exp-------------
        result = nli.wordentail_experiment(
            train_data=wordentail_data['word_disjoint']['train'],
            assess_data=wordentail_data['word_disjoint']['dev'],
            model=net,
            vector_func=lambda x: np.array([x]),
            vector_combo_func=vec_concatenate)

        return result['macro-F1']
        
    #############################################################################
    # Idea: Variations of vector_combo_func.
    def system_1():
        
        # Data------------         
        with open(wordentail_filename) as f:
            wordentail_data = json.load(f)
        
        print("Distribution of labels : \n{0}".format(pd.DataFrame(wordentail_data['word_disjoint']['train'])[1].value_counts()))

        def vec_merge(u, v):
            """Merge different feature reps including array diff, max, avg etc."""
            return np.concatenate((u, v, vec_diff(u, v), vec_max(u,v)))
        
        # Model-----------
        net = TorchShallowNeuralClassifier(hidden_dim=50, max_iter=100)
        print(net)
        
        # Exp-------------
        result = nli.wordentail_experiment(
            train_data=wordentail_data['word_disjoint']['train'],
            assess_data=wordentail_data['word_disjoint']['dev'],
            model=net,
            vector_func=glove_vec,
            vector_combo_func=vec_merge)

        return result['macro-F1']

    #######################################################################
    # Idea: Retrofit GLOVE using WordNet
    def system_2():
        
        # Data------------   
        with open(wordentail_filename) as f:
            wordentail_data = json.load(f)

        X_glove = pd.DataFrame(GLOVE).T
        print(X_glove.shape)

        def convert_edges_to_indices(edges, Q):
            lookup = dict(zip(Q.index, range(Q.shape[0])))
            index_edges = defaultdict(set)
            for start, finish_nodes in edges.items():
                s = lookup.get(start)
                if s:
                    f = {lookup[n] for n in finish_nodes if n in lookup}
                    if f:
                        index_edges[s] = f
            return index_edges

        wn_index_edges = convert_edges_to_indices(wn_edges, X_glove)
        wn_retro = Retrofitter(verbose=True)
        X_retro = wn_retro.fit(X_glove, wn_index_edges)
        print(X_retro.shape)

        def retro_vec(w):    
            """Return `w`'s Retrofitted representation if available, else return 
            a random vector."""
            return X_retro.loc[w].values if w in X_retro.index else randvec(w, n=glove_dim)

        # Model-----------
        net = TorchShallowNeuralClassifier(hidden_dim=50, max_iter=100)
        print(net)

        # Exp-------------
        result = nli.wordentail_experiment(
            train_data=wordentail_data['word_disjoint']['train'],
            assess_data=wordentail_data['word_disjoint']['dev'],
            model=net,
            vector_func=retro_vec,
            vector_combo_func=vec_concatenate)

        return result['macro-F1']
        
    ################################################################
    # Idea: Data augmentation of 'entails' class using wordnet
    def system_3():
        
        # Data------------  
        with open(wordentail_filename) as f:
            wordentail_data = json.load(f)

        x_train = wordentail_data['word_disjoint']['train']
        print("Existing distribution of labels : \n{0}".format(pd.DataFrame(x_train)[1].value_counts()))

        # get wordnet edges
        def get_wordnet_edges():
            edges = defaultdict(set)
            for ss in wn.all_synsets():
                lem_names = {lem.name() for lem in ss.lemmas()}
                for lem in lem_names:
                    edges[lem] |= lem_names
            return edges

        wn_edges = get_wordnet_edges()

        # data augmentation of positive entailments.
        positive_entailments = []
        for premise_hypothesis, label in x_train:
            if label == 1:
                positive_entailments.append(premise_hypothesis)

        print("Current count of positives: {0}".format(len(positive_entailments)))

        positive_entailments_ex = []
        for premise_hypothesis in positive_entailments:
            premise = premise_hypothesis[0]
            hypothesis = premise_hypothesis[1]

            for wn_premise in wn_edges[premise]:
                if premise == wn_premise:
                    continue
                for wn_hypothesis in wn_edges[hypothesis]:
                    if wn_hypothesis == hypothesis:
                        continue

                    positive_entailments_ex.append([wn_premise, wn_hypothesis])

        print("New count of positives to add: {0}".format(len(positive_entailments_ex)))
        x_train.extend([[item, 1] for item in positive_entailments_ex])

        print("New distribution of labels : \n{0}".format(pd.DataFrame(wordentail_data['word_disjoint']['train'])[1].value_counts()))

        # Model-----------
        net = TorchShallowNeuralClassifier(hidden_dim=50, max_iter=100)
        
        # Exp-------------
        result = nli.wordentail_experiment(
            train_data=wordentail_data['word_disjoint']['train'],
            assess_data=wordentail_data['word_disjoint']['dev'],
            model=net,
            vector_func=glove_vec,
            vector_combo_func=vec_concatenate)

        return result['macro-F1']
    
    ###################################################################
    print("System 0 (Original) Score:{0}".format(system_0_original()))
    print("="*100)
    
#     print("System 1 Score:{0}".format(system_1()))
#     print("="*100)
    
#     print("System 2 Score:{0}".format(system_2()))  
#     print("="*100)
    
#     print("System 3 Score:{0}".format(system_3()))  
#     print("="*100)
    ####################################################################
    
# Please do not remove this comment.


# ## Bake-off [1 point]
# 
# The goal of the bake-off is to achieve the highest macro-average F1 score on __word_disjoint__, on a test set that we will make available at the start of the bake-off. The announcement will go out on the discussion forum. To enter, you'll be asked to run `nli.bake_off_evaluation` on the output of your chosen `nli.wordentail_experiment` run. 
# 
# The cells below this one constitute your bake-off entry.
# 
# The rules described in the [Your original system](#Your-original-system-[3-points]) homework question are also in effect for the bake-off.
# 
# Systems that enter will receive the additional homework point, and systems that achieve the top score will receive an additional 0.5 points. We will test the top-performing systems ourselves, and only systems for which we can reproduce the reported results will win the extra 0.5 points.
# 
# Late entries will be accepted, but they cannot earn the extra 0.5 points. Similarly, you cannot win the bake-off unless your homework is submitted on time.
# 
# The announcement will include the details on where to submit your entry.

# In[ ]:


# Enter your bake-off assessment code into this cell. 
# Please do not remove this comment.
##### YOUR CODE HERE



# In[ ]:


# On an otherwise blank line in this cell, please enter
# your macro-avg f1 value as reported by the code above. 
# Please enter only a number between 0 and 1 inclusive.
# Please do not remove this comment.

##### YOUR CODE HERE

