#!/usr/bin/env python
# coding: utf-8

# # Relation extraction using distant supervision: experiments

# In[1]:


__author__ = "Bill MacCartney and Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


# ## Contents
# 
# 1. [Overview](#Overview)
# 1. [Set-up](#Set-up)
# 1. [Building a classifier](#Building-a-classifier)
#   1. [Featurizers](#Featurizers)
#   1. [Experiments](#Experiments)
# 1. [Analysis](#Analysis)
#   1. [Examining the trained models](#Examining-the-trained-models)
#   1. [Discovering new relation instances](#Discovering-new-relation-instances)

# ## Overview
# 
# OK, it's time to get (halfway) serious. Let's apply real machine learning to train a classifier on the training data, and see how it performs on the test data. We'll begin with one of the simplest machine learning setups: a bag-of-words feature representation, and a linear model trained using logistic regression.
# 
# Just like we did in the unit on [supervised sentiment analysis](https://github.com/cgpotts/cs224u/blob/master/sst_02_hand_built_features.ipynb), we'll leverage the `sklearn` library, and we'll introduce functions for featurizing instances, training models, making predictions, and evaluating results.

# ## Set-up
# 
# See [the first notebook in this unit](rel_ext_01_task.ipynb#Set-up) for set-up instructions.

# In[2]:


from collections import Counter
import os
import rel_ext
import utils


# In[3]:


# Set all the random seeds for reproducibility. Only the
# system seed is relevant for this notebook.

utils.fix_random_seeds()


# In[4]:


rel_ext_data_home = os.path.join('data', 'rel_ext_data')


# With the following steps, we build up the dataset we'll use for experiments; it unites a corpus and a knowledge base in the way we described in [the previous notebook](rel_ext_01_task.ipynb).

# In[5]:


corpus = rel_ext.Corpus(os.path.join(rel_ext_data_home, 'corpus.tsv.gz'))


# In[6]:


kb = rel_ext.KB(os.path.join(rel_ext_data_home, 'kb.tsv.gz'))


# In[7]:


dataset = rel_ext.Dataset(corpus, kb)


# The following code splits up our data in a way that supports experimentation:

# In[8]:


splits = dataset.build_splits()

splits


# ## Building a classifier

# ### Featurizers
# 
# Featurizers are functions which define the feature representation for our model. The primary input to a featurizer will be the `KBTriple` for which we are generating features. But since our features will be derived from corpus examples containing the entities of the `KBTriple`, we must also pass in a reference to a `Corpus`. And in order to make it easy to combine different featurizers, we'll also pass in a feature counter to hold the results.
# 
# Here's an implementation for a very simple bag-of-words featurizer. It finds all the corpus examples containing the two entities in the `KBTriple`, breaks the phrase appearing between the two entity mentions into words, and counts the words. Note that it makes no distinction between "forward" and "reverse" examples.

# In[9]:


def simple_bag_of_words_featurizer(kbt, corpus, feature_counter):
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        for word in ex.middle.split(' '):
            feature_counter[word] += 1
    for ex in corpus.get_examples_for_entities(kbt.obj, kbt.sbj):
        for word in ex.middle.split(' '):
            feature_counter[word] += 1
    return feature_counter


# Here's how this featurizer works on a single example:

# In[10]:


kbt = kb.kb_triples[0]

kbt


# In[11]:


corpus.get_examples_for_entities(kbt.sbj, kbt.obj)[0].middle


# In[12]:


simple_bag_of_words_featurizer(kb.kb_triples[0], corpus, Counter())


# You can experiment with adding new kinds of features just by implementing additional featurizers, following `simple_bag_of_words_featurizer` as an example.
# 
# Now, in order to apply machine learning algorithms such as those provided by `sklearn`, we need a way to convert datasets of `KBTriple`s into feature matrices. The following steps achieve that: 

# In[13]:


kbts_by_rel, labels_by_rel = dataset.build_dataset()

featurized = dataset.featurize(kbts_by_rel, featurizers=[simple_bag_of_words_featurizer])


# ### Experiments
# 
# Now we need some functions to train models, make predictions, and evaluate the results. We'll start with `train_models()`. This function takes as arguments a dictionary of data splits, a list of featurizers, the name of the split on which to train (by default, 'train'), and a model factory, which is a function which initializes an `sklearn` classifier (by default, a logistic regression classifier). It returns a dictionary holding the featurizers, the vectorizer that was used to generate the training matrix, and a dictionary holding the trained models, one per relation.

# In[14]:


train_result = rel_ext.train_models(
    splits, 
    featurizers=[simple_bag_of_words_featurizer])


# Next comes `predict()`. This function takes as arguments a dictionary of data splits, the outputs of `train_models()`, and the name of the split for which to make predictions. It returns two parallel dictionaries: one holding the predictions (grouped by relation), the other holding the true labels (again, grouped by relation).

# In[15]:


predictions, true_labels = rel_ext.predict(
    splits, train_result, split_name='dev')


# Now `evaluate_predictions()`. This function takes as arguments the parallel dictionaries of predictions and true labels produced by `predict()`. It prints summary statistics for each relation, including precision, recall, and F<sub>0.5</sub>-score, and it returns the macro-averaged F<sub>0.5</sub>-score.

# In[16]:


rel_ext.evaluate_predictions(predictions, true_labels)


# Finally, we introduce `rel_ext.experiment()`, which basically chains together `rel_ext.train_models()`, `rel_ext.predict()`, and `rel_ext.evaluate_predictions()`. For convenience, this function returns the output of `rel_ext.train_models()` as its result.

# Running `rel_ext.experiment()` in its default configuration will give us a baseline result for machine-learned models.

# In[17]:


_ = rel_ext.experiment(
    splits,
    featurizers=[simple_bag_of_words_featurizer])


# Considering how vanilla our model is, these results are quite surprisingly good! We see huge gains for every relation over our `top_k_middles_classifier` from [the previous notebook](rel_ext_01_task.ipynb#A-simple-baseline-model). This strong performance is a powerful testament to the effectiveness of even the simplest forms of machine learning.
# 
# But there is still much more we can do. To make further gains, we must not treat the model as a black box. We must open it up and get visibility into what it has learned, and more importantly, where it still falls down.

# ## Analysis

# ### Examining the trained models
# 
# One important way to gain understanding of our trained model is to inspect the model weights. What features are strong positive indicators for each relation, and what features are strong negative indicators?

# In[18]:


rel_ext.examine_model_weights(train_result)


# By and large, the high-weight features for each relation are pretty intuitive — they are words that are used to express the relation in question. (The counter-intuitive results merit a bit of investigation!)
# 
# The low-weight features (that is, features with large negative weights) may be a bit harder to understand. In some cases, however, they can be interpreted as features which indicate some _other_ relation which is anti-correlated with the target relation. (As an example, "directed" is a negative indicator for the `author` relation.)
# 
# __Optional exercise:__ Investigate one of the counter-intuitive high-weight features. Find the training examples which caused the feature to be included. Given the training data, does it make sense that this feature is a good predictor for the target relation?
# 
# <!--
# - SPOILER: Using `penalty='l1'` results in somewhat less intuitive feature weights, and about the same performance.
# - SPOILER: Using `penalty='l1', C=0.1` results in much more intuitive feature weights, but much worse performance.
# -->

# ### Discovering new relation instances
# 
# Another way to gain insight into our trained models is to use them to discover new relation instances that don't currently appear in the KB. In fact, this is the whole point of building a relation extraction system: to extend an existing KB (or build a new one) using knowledge extracted from natural language text at scale. Can the models we've trained do this effectively?
# 
# Because the goal is to discover new relation instances which are *true* but *absent from the KB*, we can't evaluate this capability automatically. But we can generate candidate KB triples and manually evaluate them for correctness.
# 
# To do this, we'll start from corpus examples containing pairs of entities which do not belong to any relation in the KB (earlier, we described these as "negative examples"). We'll then apply our trained models to each pair of entities, and sort the results by probability assigned by the model, in order to find the most likely new instances for each relation.

# In[19]:


rel_ext.find_new_relation_instances(
    dataset,
    featurizers=[simple_bag_of_words_featurizer])


# There are actually some good discoveries here! The predictions for the `author` relation seem especially good. Of course, there are also plenty of bad results, and a few that are downright comical. We may hope that as we improve our models and optimize performance in our automatic evaluations, the results we observe in this manual evaluation improve as well.
# 
# __Optional exercise:__ Note that every time we predict that a given relation holds between entities `X` and `Y`, we also predict, with equal confidence, that it holds between `Y` and `X`. Why? How could we fix this?
# 
# \[ [top](#Relation-extraction-using-distant-supervision) \]
