#!/usr/bin/env python
# coding: utf-8

# # Supervised sentiment: hand-built feature functions

# In[1]:


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


# ## Contents
# 
# 1. [Overview](#Overview)
# 1. [Set-up](#Set-up)
# 1. [Feature functions](#Feature-functions)
# 1. [Building datasets for experiments](#Building-datasets-for-experiments)
# 1. [Basic optimization](#Basic-optimization)
#   1. [Wrapper for SGDClassifier](#Wrapper-for-SGDClassifier)
#   1. [Wrapper for LogisticRegression](#Wrapper-for-LogisticRegression)
#   1. [Other scikit-learn models](#Other-scikit-learn-models)
# 1. [Experiments](#Experiments)
#   1. [Experiment with default values](#Experiment-with-default-values)
#   1. [A dev set run](#A-dev-set-run)
#   1. [Assessing BasicSGDClassifier](#Assessing-BasicSGDClassifier)
#   1. [Comparison with the baselines from Socher et al. 2013](#Comparison-with-the-baselines-from-Socher-et-al.-2013)
#   1. [A shallow neural network classifier](#A-shallow-neural-network-classifier)
#   1. [A softmax classifier in PyTorch](#A-softmax-classifier-in-PyTorch)
# 1. [Hyperparameter search](#Hyperparameter-search)
#   1. [utils.fit_classifier_with_crossvalidation](#utils.fit_classifier_with_crossvalidation)
#   1. [Example using LogisticRegression](#Example-using-LogisticRegression)
#   1. [Example using BasicSGDClassifier](#Example-using-BasicSGDClassifier)
# 1. [Statistical comparison of classifier models](#Statistical-comparison-of-classifier-models)
#   1. [Comparison with the Wilcoxon signed-rank test](#Comparison-with-the-Wilcoxon-signed-rank-test)
#   1. [Comparison with McNemar's test](#Comparison-with-McNemar's-test)

# ## Overview
# 
# * The focus of this notebook is __building feature representations__ for use with (mostly linear) classifiers (though you're encouraged to try out some non-linear ones as well!).
# 
# * The core characteristics of the feature functions we'll build here:
#    * They represent examples in __very large, very sparse feature spaces__.
#    * The individual feature functions can be __highly refined__, drawing on expert human knowledge of the domain. 
#    * Taken together, these representations don't comprehensively represent the input examples. They just identify aspects of the inputs that the classifier model can make good use of (we hope).
#    
# * These classifiers tend to be __highly competitive__. We'll look at more powerful deep learning models in the next notebook, and it will immediately become apparent that it is very difficult to get them to measure up to well-built classifiers based in sparse feature representations.

# ## Set-up
# 
# See [the previous notebook](sst_01_overview.ipynb#Set-up) for set-up instructions.

# In[2]:


from collections import Counter
import os
from sklearn.linear_model import LogisticRegression
import scipy.stats
from np_sgd_classifier import BasicSGDClassifier
import torch.nn as nn
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
import sst
import utils


# In[3]:


# Set all the random seeds for reproducibility. Only the
# system and torch seeds are relevant for this notebook.

utils.fix_random_seeds()


# In[4]:


SST_HOME = os.path.join('data', 'trees')


# ## Feature functions
# 
# * Feature representation is arguably __the most important step in any machine learning task__. As you experiment with the SST, you'll come to appreciate this fact, since your choice of feature function will have a far greater impact on the effectiveness of your models than any other choice you make.
# 
# * We will define our feature functions as `dict`s mapping feature names (which can be any object that can be a `dict` key) to their values (which must be `bool`, `int`, or `float`). 
# 
# * To prepare for optimization, we will use `sklearn`'s [DictVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) class to turn these into matrices of features. 
# 
# * The `dict`-based approach gives us a lot of flexibility and frees us from having to worry about the underlying feature matrix.

# A typical baseline or default feature representation in NLP or NLU is built from unigrams. Here, those are the leaf nodes of the tree:

# In[5]:


def unigrams_phi(tree):
    """The basis for a unigrams feature function.
    
    Parameters
    ----------
    tree : nltk.tree
        The tree to represent.
    
    Returns
    -------    
    defaultdict
        A map from strings to their counts in `tree`. (Counter maps a 
        list to a dict of counts of the elements in that list.)
    
    """
    return Counter(tree.leaves())


# In the docstring for `sst.sentiment_treebank_reader`, I pointed out that the labels on the subtrees can be used in a way that feels like cheating. Here's the most dramatic instance of this: `root_daughter_scores_phi` uses just the labels on the daughters of the root to predict the root (label). This will result in performance well north of 90% F1, but that's hardly worth reporting. (Interestingly, using the labels on the leaf nodes is much less powerful.) Anyway, don't use this function!

# In[6]:


def root_daughter_scores_phi(tree):    
    """The best way we've found to cheat without literally using the 
    labels as part of the feature representations. 
    
    Don't use this for any real experiments!
    
    """
    return Counter([child.label() for child in tree])


# It's generally good design to __write lots of atomic feature functions__ and then bring them together into a single function when running experiments. This will lead to reusable parts that you can assess independently and in sub-groups as part of development.

# ## Building datasets for experiments
# 
# The second major phase for our analysis is a kind of set-up phase. Ingredients:
# 
# * A reader like `train_reader`
# * A feature function like `unigrams_phi`
# * A class function like `binary_class_func`
# 
# The convenience function `sst.build_dataset` uses these to build a dataset for training and assessing a model. See its documentation for details on how it works. Much of this is about taking advantage of `sklearn`'s many functions for model building.

# In[7]:


train_dataset = sst.build_dataset(
    SST_HOME,
    reader=sst.train_reader,
    phi=unigrams_phi,
    class_func=sst.binary_class_func,
    vectorizer=None)


# In[9]:


train_dataset['X'].shape


# In[10]:


print("Train dataset with unigram features has {:,} examples and {:,} features".format(
        *train_dataset['X'].shape))


# Notice that `sst.build_dataset` has an optional argument `vectorizer`:
# 
# * If it is `None`, then a new vectorizer is used and returned as `dataset['vectorizer']`. This is the usual scenario when training. 
# 
# * For evaluation, one wants to represent examples exactly as they were represented during training. To ensure that this happens, pass the training `vectorizer` to this function:

# In[11]:


dev_dataset = sst.build_dataset(
    SST_HOME,
    reader=sst.dev_reader,
    phi=unigrams_phi,
    class_func=sst.binary_class_func,
    vectorizer=train_dataset['vectorizer'])


# In[12]:


print("Dev dataset with unigram features has {:,} examples "
      "and {:,} features".format(*dev_dataset['X'].shape))


# ## Basic optimization
# 
# We're now in a position to begin training supervised models!
# 
# For the most part, in this course, we will not study the theoretical aspects of machine learning optimization, concentrating instead on how to optimize systems effectively in practice. That is, this isn't a theory course, but rather an experimental, project-oriented one.
# 
# Nonetheless, we do want to avoid treating our optimizers as black boxes that work their magic and give us some assessment figures for whatever we feed into them. That seems irresponsible from a scientific and engineering perspective, and it also sends the false signal that the optimization process is inherently mysterious. So we do want to take a minute to demystify it with some simple code.
# 
# The module `np_sgd_classifier` contains a complete optimization framework, as `BasicSGDClassifier`. Well, it's complete in the sense that it achieves our full task of supervised learning. It's incomplete in the sense that it is very basic. You probably wouldn't want to use it in experiments. Rather, we're going to encourage you to rely on `sklearn` for your experiments (see below). Still, this is a good basic picture of what's happening under the hood.
# 
# So what is `BasicSGDClassifier` doing? The heart of it is the `fit` function (reflecting the usual `sklearn` naming system). This method implements a hinge-loss stochastic sub-gradient descent optimization. Intuitively, it works as follows:
# 
# 1. Start by assuming that all the feature weights are `0`.
# 1. Move through the dataset instance-by-instance in random order.
# 1. For each instance, classify it using the current weights. 
# 1. If the classification is incorrect, move the weights in the direction of the correct classification
# 
# This process repeats for a user-specified number of iterations (default `10` below), and the weight movement is tempered by a learning-rate parameter `eta` (default `0.1`). The output is a set of weights that can be used to make predictions about new (properly featurized) examples.
# 
# In more technical terms, the objective function is 
# 
# $$
#   \min_{\mathbf{w} \in \mathbb{R}^{d}}
#   \sum_{(x,y)\in\mathcal{D}} 
#   \max_{y'\in\mathbf{Y}}
#   \left[\mathbf{Score}_{\textbf{w}, \phi}(x,y') + \mathbf{cost}(y,y')\right] - \mathbf{Score}_{\textbf{w}, \phi}(x,y)
# $$
# 
# where $\mathbf{w}$ is the set of weights to be learned, $\mathcal{D}$ is the training set of example&ndash;label pairs, $\mathbf{Y}$ is the set of labels, $\mathbf{cost}(y,y') = 0$ if $y=y'$, else $1$, and $\mathbf{Score}_{\textbf{w}, \phi}(x,y')$ is the inner product of the weights 
# $\mathbf{w}$ and the example as featurized according to $\phi$.
# 
# The `fit` method is then calculating the sub-gradient of this objective. In succinct pseudo-code:
# 
# * Initialize $\mathbf{w} = \mathbf{0}$
# * Repeat $T$ times:
#     * for each $(x,y) \in \mathcal{D}$ (in random order):
#         * $\tilde{y} = \text{argmax}_{y'\in \mathcal{Y}} \mathbf{Score}_{\textbf{w}, \phi}(x,y') + \mathbf{cost}(y,y')$
#         * $\mathbf{w} =  \mathbf{w} + \eta(\phi(x,y) - \phi(x,\tilde{y}))$
#         
# This is very intuitive – push the weights in the direction of the positive cases. It doesn't require any probability theory. And such loss functions have proven highly effective in many settings. For a more powerful version of this classifier, see [sklearn.linear_model.SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier). With `loss='hinge'`, it should behave much like `BasicSGDClassifier` (but faster!).

# ### Wrapper for SGDClassifier
# 
# For the sake of our experimental framework, a simple wrapper for `SGDClassifier`:

# In[13]:


def fit_basic_sgd_classifier(X, y):    
    """Wrapper for `BasicSGDClassifier`.
    
    Parameters
    ----------
    X : 2d np.array
        The matrix of features, one example per row.        
    y : list
        The list of labels for rows in `X`.
    
    Returns
    -------
    BasicSGDClassifier
        A trained `BasicSGDClassifier` instance.
    
    """    
    mod = BasicSGDClassifier()
    mod.fit(X, y)
    return mod


# ### Wrapper for LogisticRegression
# 
# As I said above, we likely don't want to rely on `BasicSGDClassifier` (though it does a good job with SST!). Instead, we want to rely on `sklearn`. Here's a simple wrapper for [sklearn.linear.model.LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) using our 
# `build_dataset` paradigm.

# In[14]:


def fit_softmax_classifier(X, y):    
    """Wrapper for `sklearn.linear.model.LogisticRegression`. This is 
    also called a Maximum Entropy (MaxEnt) Classifier, which is more 
    fitting for the multiclass case.
    
    Parameters
    ----------
    X : 2d np.array
        The matrix of features, one example per row.
    y : list
        The list of labels for rows in `X`.
    
    Returns
    -------
    sklearn.linear.model.LogisticRegression
        A trained `LogisticRegression` instance.
    
    """
    mod = LogisticRegression(
        fit_intercept=True, 
        solver='liblinear', 
        multi_class='auto')
    mod.fit(X, y)
    return mod


# ### Other scikit-learn models
# 
# * The [sklearn.linear_model](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) package has a number of other classifier models that could be effective for SST.
# 
# * The [sklearn.ensemble](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble) package contains powerful classifiers as well. The theme that runs through all of them is that one can get better results by averaging the predictions of a bunch of more basic classifiers. A [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) will bring some of the power of deep learning models without the optimization challenges (though see [this blog post on some limitations of the current sklearn implementation](https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/)).
# 
# * The [sklearn.svm](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm) contains variations on Support Vector Machines (SVMs).

# ## Experiments
# 
# We now have all the pieces needed to run experiments. And __we're going to want to run a lot of experiments__, trying out different feature functions, taking different perspectives on the data and labels, and using different models. 
# 
# To make that process efficient and regimented, `sst` contains a function `experiment`. All it does is pull together these pieces and use them for training and assessment. It's complicated, but the flexibility will turn out to be an asset.

# ### Experiment with default values

# In[15]:


_ = sst.experiment(
    SST_HOME,
    unigrams_phi,
    fit_softmax_classifier,
    train_reader=sst.train_reader, 
    assess_reader=None, 
    train_size=0.7,
    class_func=sst.ternary_class_func,
    score_func=utils.safe_macro_f1,
    verbose=True)


# A few notes on this function call:
#     
# * Since `assess_reader=None`, the function reports performance on a random train–test split. Give `sst.dev_reader` as the argument to assess against the `dev` set.
# 
# * `unigrams_phi` is the function we defined above. By changing/expanding this function, you can start to improve on the above baseline, perhaps periodically seeing how you do on the dev set.
# 
# * `fit_softmax_classifier` is the wrapper we defined above. To assess new models, simply define more functions like this one. Such functions just need to consume an `(X, y)` constituting a dataset and return a model.

# ### A dev set run

# In[14]:


_ = sst.experiment(
    SST_HOME,
    unigrams_phi,
    fit_softmax_classifier,
    class_func=sst.ternary_class_func,
    assess_reader=sst.dev_reader)


# ### Assessing BasicSGDClassifier

# In[15]:


_ = sst.experiment(
    SST_HOME,
    unigrams_phi,
    fit_basic_sgd_classifier,
    class_func=sst.ternary_class_func,
    assess_reader=sst.dev_reader)


# ### Comparison with the baselines from Socher et al. 2013
# 
# Where does our default set-up sit with regard to published baselines for the binary problem? (Compare  [Socher et al., Table 1](http://www.aclweb.org/anthology/D/D13/D13-1170.pdf).)

# In[16]:


_ = sst.experiment(
    SST_HOME,
    unigrams_phi,
    fit_softmax_classifier,
    class_func=sst.binary_class_func,
    assess_reader=sst.dev_reader)


# ### A shallow neural network classifier
# 
# While we're at it, we might as well see whether adding a hidden layer to our softmax classifier yields any benefits. Whereas `LogisticRegression` is, at its core, computing
# 
# $$\begin{align*}
# y &= \textbf{softmax}(xW_{xy} + b_{y})
# \end{align*}$$
# 
# the shallow neural network inserts a hidden layer with a non-linear activation applied to it:
# 
# $$\begin{align*}
# h &= \tanh(xW_{xh} + b_{h}) \\
# y &= \textbf{softmax}(hW_{hy} + b_{y})
# \end{align*}$$

# In[17]:


def fit_nn_classifier(X, y):
    mod = TorchShallowNeuralClassifier(
        hidden_dim=50, max_iter=100)
    mod.fit(X, y)
    return mod


# In[18]:


_ = sst.experiment(
    SST_HOME,
    unigrams_phi, 
    fit_nn_classifier, 
    class_func=sst.binary_class_func)


# It looks like, with enough iterations (and perhaps some fiddling with the activation function and hidden dimensionality), this classifier would meet or exceed the baseline set up by `LogisticRegression`.

# ### A softmax classifier in PyTorch
# 
# Our PyTorch modules should support easy modification. For example, to turn `TorchShallowNeuralClassifier` into a `TorchSoftmaxClassifier`, one need only write a new `define_graph` method:

# In[19]:


class TorchSoftmaxClassifier(TorchShallowNeuralClassifier):
    
    def define_graph(self):
        return nn.Linear(self.input_dim, self.n_classes_)


# In[20]:


def fit_torch_softmax(X, y):
    mod = TorchSoftmaxClassifier(max_iter=100)
    mod.fit(X, y)
    return mod


# In[21]:


_ = sst.experiment(
    SST_HOME,
    unigrams_phi, 
    fit_torch_softmax, 
    class_func=sst.binary_class_func)


# ## Hyperparameter search
# 
# The training process learns __parameters__ &mdash; the weights. There are typically lots of other parameters that need to be set. For instance, our `BasicSGDClassifier` has a learning rate parameter and a training iteration parameter. These are called __hyperparameters__. The more powerful `sklearn` classifiers often have many more such hyperparameters. These are outside of the explicitly stated objective, hence the "hyper" part. 
# 
# So far, we have just set the hyperparameters by hand. However, their optimal values can vary widely between datasets, and choices here can dramatically impact performance, so we would like to set them as part of the overall experimental framework.

# ### utils.fit_classifier_with_crossvalidation
# 
# Luckily, `sklearn` provides a lot of functionality for setting hyperparameters via cross-validation. The function `utils.fit_classifier_with_crossvalidation` implements a basic framework for taking advantage of these options. 
# 
# This method has the same basic shape as `fit_softmax_classifier` above: it takes a dataset as input and returns a trained model. However, to find its favored model, it explores a space of hyperparameters supplied by the user, seeking the optimal combination of settings.
# 
# __Note__: this kind of search seems not to have a large impact for SST as we're using it. However, it can matter a lot for other data sets, and it's also an important step to take when trying to publish, since __reviewers are likely to want to check that your comparisons aren't based in part on opportunistic or ill-considered choices for the hyperparameters__.

# ### Example using LogisticRegression
# 
# Here's a fairly full-featured use of the above for the `LogisticRegression` model family:

# In[22]:


def fit_softmax_with_crossvalidation(X, y):
    """A MaxEnt model of dataset with hyperparameter 
    cross-validation. Some notes:
        
    * 'fit_intercept': whether to include the class bias feature.
    * 'C': weight for the regularization term (smaller is more regularized).
    * 'penalty': type of regularization -- roughly, 'l1' ecourages small 
      sparse models, and 'l2' encourages the weights to conform to a 
      gaussian prior distribution.
    
    Other arguments can be cross-validated; see 
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    
    Parameters
    ----------
    X : 2d np.array
        The matrix of features, one example per row.
        
    y : list
        The list of labels for rows in `X`.   
    
    Returns
    -------
    sklearn.linear_model.LogisticRegression
        A trained model instance, the best model found.
    
    """    
    basemod = LogisticRegression(
        fit_intercept=True, 
        solver='liblinear', 
        multi_class='auto')
    cv = 5
    param_grid = {'fit_intercept': [True, False], 
                  'C': [0.4, 0.6, 0.8, 1.0, 2.0, 3.0],
                  'penalty': ['l1','l2']}    
    best_mod = utils.fit_classifier_with_crossvalidation(
        X, y, basemod, cv, param_grid)
    return best_mod


# In[23]:


softmax_experiment = sst.experiment(
    SST_HOME,
    unigrams_phi,
    fit_softmax_with_crossvalidation, 
    class_func=sst.ternary_class_func)


# ### Example using BasicSGDClassifier

# The models written for this course are also compatible with this framework. They ["duck type"](https://en.wikipedia.org/wiki/Duck_typing) the `sklearn` models by having methods `fit`, `predict`, `get_params`, and `set_params`, and an attribute `params`.

# In[24]:


def fit_basic_sgd_classifier_with_crossvalidation(X, y):
    basemod = BasicSGDClassifier()
    cv = 5
    param_grid = {'eta': [0.01, 0.1, 1.0], 'max_iter': [10]}
    best_mod = utils.fit_classifier_with_crossvalidation(
        X, y, basemod, cv, param_grid)
    return best_mod


# In[25]:


sgd_experiment = sst.experiment(
    SST_HOME,
    unigrams_phi,
    fit_basic_sgd_classifier_with_crossvalidation, 
    class_func=sst.ternary_class_func)


# ## Statistical comparison of classifier models
# 
# Suppose two classifiers differ according to an effectiveness measure like F1 or accuracy. Are they meaningfully different?
# 
# * For very large datasets, the answer might be clear: if performance is very stable across different train/assess splits and the difference in terms of correct predictions has practical importance, then you can clearly say yes. 
# 
# * With smaller datasets, or models whose performance is closer together, it can be harder to determine whether the two models are different. We can address this question in a basic way with repeated runs and basic null-hypothesis testing on the resulting score vectors.
# 
# In general, one wants to compare __two feature functions against the same model__, or one wants to compare __two models with the same feature function used for both__. If both are changed at the same time, then it will be hard to figure out what is causing any differences you see.

# ### Comparison with the Wilcoxon signed-rank test
# 
# The function `sst.compare_models` is designed for such testing. The default set-up uses the non-parametric [Wilcoxon signed-rank test](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test) to make the comparisons, which is relatively conservative and recommended by [Demšar 2006](http://www.jmlr.org/papers/v7/demsar06a.html) for cases where one can afford to do multiple assessments. For discussion, see [the evaluation methods notebook](evaluation_methods.ipynb#Wilcoxon-signed-rank-test).
# 
# Here's an example showing the default parameters values and comparing `LogisticRegression` and `BasicSGDClassifier`:

# In[26]:


_ = sst.compare_models(
    SST_HOME,
    unigrams_phi,
    fit_softmax_classifier,
    stats_test=scipy.stats.wilcoxon,
    trials=10,
    phi2=None,  # Defaults to same as first required argument.
    train_func2=fit_basic_sgd_classifier, # Defaults to same as second required argument.
    reader=sst.train_reader, 
    train_size=0.7, 
    class_func=sst.ternary_class_func, 
    score_func=utils.safe_macro_f1)


# ### Comparison with McNemar's test
# 
# [McNemar's test](https://en.wikipedia.org/wiki/McNemar%27s_test) operates directly on the vectors of predictions for the two models being compared. As such, it doesn't require repeated runs, which is good where optimization is expensive. For discussion, see [the evaluation methods notebook](evaluation_methods.ipynb#McNemar's-test).

# In[27]:


m = utils.mcnemar(
    softmax_experiment['assess_dataset']['y'], 
    sgd_experiment['predictions'],
    softmax_experiment['predictions'])


# In[28]:


p = "p < 0.0001" if m[1] < 0.0001 else m[1]

print("McNemar's test: {0:0.02f} ({1:})".format(m[0], p))

