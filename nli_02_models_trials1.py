#!/usr/bin/env python
# coding: utf-8

# # Natural language inference: models

# In[1]:


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


# ## Contents
# 
# 1. [Contents](#Contents)
# 1. [Overview](#Overview)
# 1. [Set-up](#Set-up)
# 1. [Sparse feature representations](#Sparse-feature-representations)
#   1. [Feature representations](#Feature-representations)
#   1. [Model wrapper](#Model-wrapper)
#   1. [Assessment](#Assessment)
# 1. [Sentence-encoding models](#Sentence-encoding-models)
#   1. [Dense representations with a linear classifier](#Dense-representations-with-a-linear-classifier)
#   1. [Dense representations with a shallow neural network](#Dense-representations-with-a-shallow-neural-network)
#   1. [Sentence-encoding RNNs](#Sentence-encoding-RNNs)
#   1. [Other sentence-encoding model ideas](#Other-sentence-encoding-model-ideas)
# 1. [Chained models](#Chained-models)
#   1. [Simple RNN](#Simple-RNN)
#   1. [Separate premise and hypothesis RNNs](#Separate-premise-and-hypothesis-RNNs)
# 1. [Attention mechanisms](#Attention-mechanisms)
# 1. [Error analysis with the MultiNLI annotations](#Error-analysis-with-the-MultiNLI-annotations)
# 1. [Other findings](#Other-findings)
# 1. [Exploratory exercises](#Exploratory-exercises)

# ## Overview
# 
# This notebook defines and explores a number of models for NLI. The general plot is familiar from [our work with the Stanford Sentiment Treebank](sst_01_overview.ipynb):
# 
# 1. Models based on sparse feature representations
# 1. Linear classifiers and feed-forward neural classifiers using dense feature representations
# 1. Recurrent and tree-structured neural networks
# 
# The twist here is that, while NLI is another classification problem, the inputs have important high-level structure: __a premise__ and __a hypothesis__. This invites exploration of a host of neural model designs:
# 
# * In __sentence-encoding__ models, the premise and hypothesis are analyzed separately, combined only for the final classification step.
# 
# * In __chained__ models, the premise is processed first, then the hypotheses, giving a unified representation of the pair.
# 
# NLI resembles sequence-to-sequence problems like __machine translation__ and __language modeling__. The central modeling difference is that NLI doesn't produce an output sequence, but rather consumes two sequences to produce a label. Still, there are enough affinities that many ideas have been shared among these fields.

# ## Set-up
# 
# See [the previous notebook](nli_01_task_and_data.ipynb#Set-up) for set-up instructions for this unit. 

# In[2]:


from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.utils.data
from torch_model_base import TorchModelBase
from torch_rnn_classifier import TorchRNNClassifier, TorchRNNClassifierModel
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from torch_rnn_classifier import TorchRNNClassifier
import nli
import os
import utils


# In[3]:


# Set all the random seeds for reproducibility. Only the
# system and torch seeds are relevant for this notebook.

utils.fix_random_seeds()


# In[4]:


GLOVE_HOME = os.path.join('data', 'glove.6B')

DATA_HOME = os.path.join("data", "nlidata")

SNLI_HOME = os.path.join(DATA_HOME, "snli_1.0")

MULTINLI_HOME = os.path.join(DATA_HOME, "multinli_1.0")

ANNOTATIONS_HOME = os.path.join(DATA_HOME, "multinli_1.0_annotations")


# ## Sparse feature representations
# 
# We begin by looking at models based in sparse, hand-built feature representations. As in earlier units of the course, we will see that __these models are competitive__: easy to design, fast to optimize, and highly effective.

# ### Feature representations
# 
# The guiding idea for NLI sparse features is that one wants to knit together the premise and hypothesis, so that the model can learn about their relationships rather than just about each part separately.

# With `word_overlap_phi`, we just get the set of words that occur in both the premise and hypothesis.

# In[5]:


def word_overlap_phi(t1, t2):    
    """Basis for features for the words in both the premise and hypothesis.
    This tends to produce very sparse representations.
    
    Parameters
    ----------
    t1, t2 : `nltk.tree.Tree`
        As given by `str2tree`.
        
    Returns
    -------
    defaultdict
       Maps each word in both `t1` and `t2` to 1.
    
    """
    overlap = set([w1 for w1 in t1.leaves() if w1 in t2.leaves()])
    return Counter(overlap)


# With `word_cross_product_phi`, we count all the pairs $(w_{1}, w_{2})$ where $w_{1}$ is a word from the premise and $w_{2}$ is a word from the hypothesis. This creates a very large feature space. These models are very strong right out of the box, and they can be supplemented with more fine-grained features.

# In[6]:


def word_cross_product_phi(t1, t2):
    """Basis for cross-product features. This tends to produce pretty 
    dense representations.
    
    Parameters
    ----------
    t1, t2 : `nltk.tree.Tree`
        As given by `str2tree`.
        
    Returns
    -------
    defaultdict
        Maps each (w1, w2) in the cross-product of `t1.leaves()` and 
        `t2.leaves()` to its count. This is a multi-set cross-product
        (repetitions matter).
    
    """
    return Counter([(w1, w2) for w1, w2 in product(t1.leaves(), t2.leaves())])


# ### Model wrapper
# 
# Our experiment framework is basically the same as the one we used for the Stanford Sentiment Treebank. Here, I actually use `utils.fit_classifier_with_crossvalidation` (introduced in that unit) to create a wrapper around `LogisticRegression` for cross-validation of hyperparameters. At this point, I am not sure what parameters will be good for our NLI datasets, so this hyperparameter search is vital.

# In[7]:


def fit_softmax_with_crossvalidation(X, y):
    """A MaxEnt model of dataset with hyperparameter cross-validation.
    
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
    cv = 3
    param_grid = {'C': [0.4, 0.6, 0.8, 1.0],
                  'penalty': ['l1','l2']}    
    best_mod = utils.fit_classifier_with_crossvalidation(
        X, y, basemod, cv, param_grid)
    return best_mod


# ### Assessment
# 
# Because SNLI and MultiNLI are huge, we can't afford to do experiments on the full datasets all the time. Thus, we will mainly work within the training sets, using the train readers to sample smaller datasets that can then be divided for training and assessment.
# 
# Here, we sample 10% of the training examples. I set the random seed (`random_state=42`) so that we get consistency across the samples; setting `random_state=None` will give new random samples each time.

# In[8]:


train_reader = nli.SNLITrainReader(
    SNLI_HOME, samp_percentage=0.10, random_state=42)


# An experimental dataset can be built directly from the reader and a feature function:

# In[9]:


dataset = nli.build_dataset(train_reader, word_overlap_phi)


# In[10]:


dataset.keys()


# However, it's more efficient to use `nli.experiment` to bring all these pieces together. This wrapper will work for all the models we consider.

# In[11]:


# _ = nli.experiment(
#     train_reader=nli.SNLITrainReader(
#         SNLI_HOME, samp_percentage=0.10, random_state=42),
#     phi=word_overlap_phi,
#     train_func=fit_softmax_with_crossvalidation,
#     assess_reader=None,
#     random_state=42)


# In[12]:


# _ = nli.experiment(
#     train_reader=nli.SNLITrainReader(
#         SNLI_HOME, samp_percentage=0.10, random_state=42),
#     phi=word_cross_product_phi,
#     train_func=fit_softmax_with_crossvalidation,
#     assess_reader=None,
#     random_state=42)


# As expected `word_cross_product_phi` is very strong. At this point, one might consider scaling up to `samp_percentage=None`, i.e., the full training set. Such a baseline is very similar to the one established in [the original SNLI paper by Bowman et al.](https://aclanthology.info/papers/D15-1075/d15-1075) for models like this one.

# ## Sentence-encoding models
# 
# We turn now to sentence-encoding models. The hallmark of these is that the premise and hypothesis get their own representation in some sense, and then those representations are combined to predict the label. [Bowman et al. 2015](http://aclweb.org/anthology/D/D15/D15-1075.pdf) explore models of this form as part of introducing SNLI.
# 
# The feed-forward networks we used in [the word-level bake-off](nli_wordentail_bakeoff.ipynb) are members of this family of models: each word was represented separately, and the concatenation of those representations was used as the input to the model.

# ### Dense representations with a linear classifier
# 
# Perhaps the simplest sentence-encoding model sums (or averages, etc.) the word representations for the premise, does the same for the hypothesis, and concatenates those two representations for use as the input to a linear classifier. 
# 
# Here's a diagram that is meant to suggest the full space of models of this form:
# 
# <img src="fig/nli-softmax.png" width=800 />

# Here's an implementation of this model where 
# 
# * The embedding is GloVe.
# * The word representations are summed.
# * The premise and hypothesis vectors are concatenated.
# * A softmax classifier is used at the top.

# In[13]:


glove_lookup = utils.glove2dict(
    os.path.join(GLOVE_HOME, 'glove.6B.50d.txt'))


# In[14]:


def glove_leaves_phi(t1, t2, np_func=np.sum):
    """Represent `tree` as a combination of the vector of its words.
    
    Parameters
    ----------
    t1 : nltk.Tree   
    t2 : nltk.Tree   
    np_func : function (default: np.sum)
        A numpy matrix operation that can be applied columnwise, 
        like `np.mean`, `np.sum`, or `np.prod`. The requirement is that 
        the function take `axis=0` as one of its arguments (to ensure
        columnwise combination) and that it return a vector of a 
        fixed length, no matter what the size of the tree is.
    
    Returns
    -------
    np.array
            
    """    
    prem_vecs = _get_tree_vecs(t1, glove_lookup, np_func)  
    hyp_vecs = _get_tree_vecs(t2, glove_lookup, np_func)  
    return np.concatenate((prem_vecs, hyp_vecs))
    
    
def _get_tree_vecs(tree, lookup, np_func):
    allvecs = np.array([lookup[w] for w in tree.leaves() if w in lookup])    
    if len(allvecs) == 0:
        dim = len(next(iter(lookup.values())))
        feats = np.zeros(dim)    
    else:       
        feats = np_func(allvecs, axis=0)      
    return feats


# In[15]:


# _ = nli.experiment(
#     train_reader=nli.SNLITrainReader(
#         SNLI_HOME, samp_percentage=0.10, random_state=42),
#     phi=glove_leaves_phi,
#     train_func=fit_softmax_with_crossvalidation,
#     assess_reader=None,
#     random_state=42,
#     vectorize=False)  # Ask `experiment` not to featurize; we did it already.


# ### Dense representations with a shallow neural network
# 
# A small tweak to the above is to use a neural network instead of a softmax classifier at the top:

# In[16]:


def fit_shallow_neural_classifier_with_crossvalidation(X, y):    
    basemod = TorchShallowNeuralClassifier(max_iter=50)
    cv = 3
    param_grid = {'hidden_dim': [25, 50, 100]}
    best_mod = utils.fit_classifier_with_crossvalidation(
        X, y, basemod, cv, param_grid)
    return best_mod


# In[17]:


# _ = nli.experiment(
#     train_reader=nli.SNLITrainReader(
#         SNLI_HOME, samp_percentage=0.10, random_state=42),
#     phi=glove_leaves_phi,
#     train_func=fit_shallow_neural_classifier_with_crossvalidation,
#     assess_reader=None,
#     random_state=42,
#     vectorize=False)  # Ask `experiment` not to featurize; we did it already.


# ### Sentence-encoding RNNs
# 
# A more sophisticated sentence-encoding model processes the premise and hypothesis with separate RNNs and uses the concatenation of their final states as the basis for the classification decision at the top:
# 
# <img src="fig/nli-rnn-sentencerep.png" width=800 />

# It is relatively straightforward to extend `torch_rnn_classifier` so that it can handle this architecture:

# #### A sentence-encoding dataset
# 
# Whereas `torch_rnn_classifier.TorchRNNDataset` creates batches that consist of `(sequence, sequence_length, label)` triples, the sentence encoding model requires us to double the first two components. The most important features of this is `collate_fn`, which determines what the batches look like:

# In[18]:


class TorchRNNSentenceEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, seq_lengths, y):
        self.prem_seqs, self.hyp_seqs = sequences
        self.prem_lengths, self.hyp_lengths = seq_lengths
        self.y = y
        assert len(self.prem_seqs) == len(self.y)

    @staticmethod
    def collate_fn(batch):
        X_prem, X_hyp, prem_lengths, hyp_lengths, y = zip(*batch)
        prem_lengths = torch.LongTensor(prem_lengths)
        hyp_lengths = torch.LongTensor(hyp_lengths)
        y = torch.LongTensor(y)
        return (X_prem, X_hyp), (prem_lengths, hyp_lengths), y

    def __len__(self):
        return len(self.prem_seqs)

    def __getitem__(self, idx):
        return (self.prem_seqs[idx], self.hyp_seqs[idx],
                self.prem_lengths[idx], self.hyp_lengths[idx],
                self.y[idx])


# #### A sentence-encoding model
# 
# With `TorchRNNSentenceEncoderClassifierModel`, we subclass `torch_rnn_classifier.TorchRNNClassifierModel` and make use of many of its parameters. The changes:
# 
# * We add an attribute `self.hypothesis_rnn` for encoding the hypothesis. (The super class has `self.rnn`, which we use for the premise.)
# * The `forward` method concatenates the final states from the premise and hypothesis, and they are the input to the classifier layer, which is unchanged from before but how accepts inputs that are double the size.

# In[19]:


class TorchRNNSentenceEncoderClassifierModel(TorchRNNClassifierModel):
    def __init__(self, vocab_size, embed_dim, embedding, use_embedding,
            hidden_dim, output_dim, bidirectional, device):
        super(TorchRNNSentenceEncoderClassifierModel, self).__init__(
            vocab_size, embed_dim, embedding, use_embedding,
            hidden_dim, output_dim, bidirectional, device)
        self.hypothesis_rnn = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=self.bidirectional)
        if bidirectional:
            classifier_dim = hidden_dim * 2 * 2
        else:
            classifier_dim = hidden_dim * 2
        self.classifier_layer = nn.Linear(
            classifier_dim, output_dim)

    def forward(self, X, seq_lengths):
        X_prem, X_hyp = X
        prem_lengths, hyp_lengths = seq_lengths
        prem_state = self.rnn_forward(X_prem, prem_lengths, self.rnn)
        hyp_state = self.rnn_forward(X_hyp, hyp_lengths, self.hypothesis_rnn)
        state = torch.cat((prem_state, hyp_state), dim=1)
        logits = self.classifier_layer(state)
        return logits


# #### A sentence-encoding model interface
# 
# Finally, we subclass `TorchRNNClassifier`. Here, just need to redefine three methods: `build_dataset` and `build_graph` to make use of the new components above, and `predict_proba` so that it deals with the premise/hypothesis shape of new inputs.

# In[20]:


class TorchRNNSentenceEncoderClassifier(TorchRNNClassifier):

    def build_dataset(self, X, y):
        X_prem, X_hyp = zip(*X)
        X_prem, prem_lengths = self._prepare_dataset(X_prem)
        X_hyp, hyp_lengths = self._prepare_dataset(X_hyp)
        return TorchRNNSentenceEncoderDataset(
            (X_prem, X_hyp), (prem_lengths, hyp_lengths), y)

    def build_graph(self):
        return TorchRNNSentenceEncoderClassifierModel(
            len(self.vocab),
            embedding=self.embedding,
            embed_dim=self.embed_dim,
            use_embedding=self.use_embedding,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_classes_,
            bidirectional=self.bidirectional,
            device=self.device)

    def predict_proba(self, X):
        with torch.no_grad():
            X_prem, X_hyp = zip(*X)
            X_prem, prem_lengths = self._prepare_dataset(X_prem)
            X_hyp, hyp_lengths = self._prepare_dataset(X_hyp)
            preds = self.model((X_prem, X_hyp), (prem_lengths, hyp_lengths))
            preds = torch.softmax(preds, dim=1).cpu().numpy()
            return preds


# #### Simple example
# 
# This toy problem illustrates how this works in detail:

# In[21]:


def simple_example():
    vocab = ['a', 'b', '$UNK']

    # Reversals are good, and other pairs are bad:
    train = [
        [(list('ab'), list('ba')), 'good'],
        [(list('aab'), list('baa')), 'good'],
        [(list('abb'), list('bba')), 'good'],
        [(list('aabb'), list('bbaa')), 'good'],
        [(list('ba'), list('ba')), 'bad'],
        [(list('baa'), list('baa')), 'bad'],
        [(list('bba'), list('bab')), 'bad'],
        [(list('bbaa'), list('bbab')), 'bad'],
        [(list('aba'), list('bab')), 'bad']]

    test = [
        [(list('baaa'), list('aabb')), 'bad'],
        [(list('abaa'), list('baaa')), 'bad'],
        [(list('bbaa'), list('bbaa')), 'bad'],
        [(list('aaab'), list('baaa')), 'good'],
        [(list('aaabb'), list('bbaaa')), 'good']]

    mod = TorchRNNSentenceEncoderClassifier(
        vocab,
        max_iter=100,
        embed_dim=50,
        hidden_dim=50)

    X, y = zip(*train)
    mod.fit(X, y)

    X_test, y_test = zip(*test)
    preds = mod.predict(X_test)

    print("\nPredictions:")
    for ex, pred, gold in zip(X_test, preds, y_test):
        score = "correct" if pred == gold else "incorrect"
        print("{0:>6} {1:>6} - predicted: {2:>4}; actual: {3:>4} - {4}".format(
            "".join(ex[0]), "".join(ex[1]), pred, gold, score))


# In[22]:


simple_example()


# #### Example SNLI run

# In[23]:


def sentence_encoding_rnn_phi(t1, t2):
    """Map `t1` and `t2` to a pair of lits of leaf nodes."""
    return (t1.leaves(), t2.leaves())


# In[24]:


def get_sentence_encoding_vocab(X, n_words=None):    
    wc = Counter([w for pair in X for ex in pair for w in ex])
    wc = wc.most_common(n_words) if n_words else wc.items()
    vocab = {w for w, c in wc}
    vocab.add("$UNK")
    return sorted(vocab)


# In[25]:


def fit_sentence_encoding_rnn(X, y):   
    vocab = get_sentence_encoding_vocab(X, n_words=10000)
    mod = TorchRNNSentenceEncoderClassifier(
        vocab, hidden_dim=50, max_iter=10)
    mod.fit(X, y)
    return mod


# In[26]:


_ = nli.experiment(
    train_reader=nli.SNLITrainReader(
        SNLI_HOME, samp_percentage=0.10, random_state=42), 
    phi=sentence_encoding_rnn_phi,
    train_func=fit_sentence_encoding_rnn,
    assess_reader=None,
    random_state=42,
    vectorize=False)


# ### Other sentence-encoding model ideas
# 
# Given that [we already explored tree-structured neural networks (TreeNNs)](sst_03_neural_networks.ipynb#Tree-structured-neural-networks), it's natural to consider these as the basis for sentence-encoding NLI models:
# 
# <img src="fig/nli-treenn.png" width=800 />
# 
# And this is just the begnning: any model used to represent sentences is presumably a candidate for use in sentence-encoding NLI!

# ## Chained models
# 
# The final major class of NLI designs we look at are those in which the premise and hypothesis are processed sequentially, as a pair. These don't deliver representations of the premise or hypothesis separately. They bear the strongest resemblance to classic sequence-to-sequence models.

# ### Simple RNN
# 
# In the simplest version of this model, we just concatenate the premise and hypothesis. The model itself is identical to the one we used for the Stanford Sentiment Treebank:
# 
# <img src="fig/nli-rnn-chained.png" width=800 />

# To implement this, we can use `TorchRNNClassifier` out of the box. We just need to concatenate the leaves of the premise and hypothesis trees:

# In[27]:


def simple_chained_rep_rnn_phi(t1, t2):
    """Map `t1` and `t2` to a single list of leaf nodes.
    
    A slight variant might insert a designated boundary symbol between 
    the premise leaves and the hypothesis leaves. Be sure to add it to 
    the vocab in that case, else it will be $UNK.
    """
    return t1.leaves() + t2.leaves()


# Here's a quick evaluation, just to get a feel for this model:

# In[28]:


def fit_simple_chained_rnn(X, y):   
    vocab = utils.get_vocab(X, n_words=10000)
    mod = TorchRNNClassifier(vocab, hidden_dim=50, max_iter=10)
    mod.fit(X, y)
    return mod


# In[29]:


_ = nli.experiment(
    train_reader=nli.SNLITrainReader(
        SNLI_HOME, samp_percentage=0.10, random_state=42), 
    phi=simple_chained_rep_rnn_phi,
    train_func=fit_simple_chained_rnn,
    assess_reader=None,
    random_state=42,
    vectorize=False)


# ### Separate premise and hypothesis RNNs
# 
# A natural variation on the above is to give the premise and hypothesis each their own RNN:
# 
# <img src="fig/nli-rnn-chained-separate.png" width=800 />
# 
# This greatly increases the number of parameters, but it gives the model more chances to learn that appearing in the premise is different from appearing in the hypothesis. One could even push this idea further by giving the premise and hypothesis their own embeddings as well. One could implement this easily by modifying [the sentence-encoder version defined above](#Sentence-encoding-RNNs).

# ## Attention mechanisms
# 
# Many of the best-performing systems in [the SNLI leaderboard](https://nlp.stanford.edu/projects/snli/) use __attention mechanisms__ to help the model learn important associations between words in the premise and words in the hypothesis. I believe [Rocktäschel et al. (2015)](https://arxiv.org/pdf/1509.06664v1.pdf) were the first to explore such models for NLI.
# 
# For instance, if _puppy_ appears in the premise and _dog_ in the conclusion, then that might be a high-precision indicator that the correct relationship is entailment.
# 
# This diagram is a high-level schematic for adding attention mechanisms to a chained RNN model for NLI:
# 
# <img src="fig/nli-rnn-attention.png" width=800 />
# 
# Since TensorFlow will handle the details of backpropagation, implementing these models is largely reduced to figuring out how to wrangle the states of the model in the desired way.

# ## Error analysis with the MultiNLI annotations
# 
# The annotations included with the MultiNLI corpus create some powerful yet easy opportunities for error analysis right out of the box. This section illustrates how to make use of them with models you've trained.
# 
# First, we train a sentence-encoding model on a sample of the MultiNLI data, just for illustrative purposes:

# In[30]:


rnn_multinli_experiment = nli.experiment(
    train_reader=nli.MultiNLITrainReader(
        MULTINLI_HOME, samp_percentage=0.10, random_state=42), 
    phi=sentence_encoding_rnn_phi,
    train_func=fit_sentence_encoding_rnn,
    assess_reader=None,
    random_state=42,
    vectorize=False)


# The return value of `nli.experiment` contains the information we need to make predictions on new examples. 
# 
# Next, we load in the 'matched' condition annotations ('mismatched' would work as well):

# In[31]:


matched_ann_filename = os.path.join(
    ANNOTATIONS_HOME,
    "multinli_1.0_matched_annotations.txt")


# In[32]:


matched_ann = nli.read_annotated_subset(
    matched_ann_filename, MULTINLI_HOME)


# The following function uses `rnn_multinli_experiment` to make predictions on annotated examples, and harvests some other information that is useful for error analysis:

# In[33]:


def predict_annotated_example(ann, experiment_results):
    model = experiment_results['model']
    phi = experiment_results['phi']
    ex = ann['example']
    prem = ex.sentence1_parse
    hyp = ex.sentence2_parse
    feats = phi(prem, hyp)
    pred = model.predict([feats])[0]
    gold = ex.gold_label
    data = {cat: True for cat in ann['annotations']}
    data.update({'gold': gold, 'prediction': pred, 'correct': gold == pred})
    return data


# Finally, this function applies `predict_annotated_example` to a collection of annotated examples and puts the results in a `pd.DataFrame` for flexible analysis:

# In[34]:


def get_predictions_for_annotated_data(anns, experiment_results):
    data = []
    for ex_id, ann in anns.items():
        results = predict_annotated_example(ann, experiment_results)
        data.append(results)
    return pd.DataFrame(data)


# In[35]:


ann_analysis_df = get_predictions_for_annotated_data(
    matched_ann, rnn_multinli_experiment)


# With `ann_analysis_df`, we can see how the model does on individual annotation categories:

# In[36]:


pd.crosstab(ann_analysis_df['correct'], ann_analysis_df['#MODAL'])


# ## Other findings
# 
# 1. A high-level lesson of [the SNLI leaderboard](https://nlp.stanford.edu/projects/snli/) is that one can do __extremely well__ with simple neural models whose hyperparameters are selected via extensive cross-validation. This is mathematically interesting but might be dispiriting to those of us without vast resources to devote to these computations! (On the flip side, cleverly designed linear models or ensembles with sparse feature representations might beat all of these entrants with a fraction of the computational budget.)
# 
# 1. In an outstanding project for this course in 2016, [Leonid Keselman](https://leonidk.com) observed that [one can do much better than chance on SNLI by processing only the hypothesis](https://leonidk.com/stanford/cs224u.html). This relates to [observations we made in the word-level homework/bake-off](hw4_wordentail.ipynb) about how certain terms will tend to appear more on the right in entailment pairs than on the left. Last year, a number of groups independently (re-)discovered this fact and published analyses: [Poliak et al. 2018](https://aclanthology.info/papers/S18-2023/s18-2023), [Tsuchiya 2018](https://aclanthology.info/papers/L18-1239/l18-1239), [Gururangan  et al. 2018](https://aclanthology.info/papers/N18-2017/n18-2017).
# 
# 1. As we pointed out at the start of this unit, [Dagan et al. (2006) pitched NLI as a general-purpose NLU task](nli_01_task_and_data.ipynb#Overview). We might then hope that the representations we learn on this task will transfer to others. So far, the evidence for this is decidedly mixed. I suspect the core scientific idea is sound, but that __we still lack the needed methods for doing transfer learning__.
# 
# 1. For SNLI, we seem to have entered the inevitable phase in machine learning problems where __ensembles do best__.

# ## Exploratory exercises
# 
# These are largely meant to give you a feel for the material, but some of them could lead to projects and help you with future work for the course. These are not for credit.
# 
# 1. When we [feed dense representations to a simple classifier](#Dense-representations-with-a-linear-classifier), what is the effect of changing the combination functions (e.g., changing `sum` to `mean`; changing `concatenate` to `difference`)? What happens if we swap out `LogisticRegression` for, say, an [sklearn.ensemble.RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) instance?
# 
# 1. Implement the [Separate premise and hypothesis RNN](#Separate-premise-and-hypothesis-RNNs) and evaluate it, comparing in particular against [the version that simply concatenates the premise and hypothesis](#Simple-RNN). Does having all these additional parameters pay off? Do you need more training examples to start to see the value of this idea?
# 
# 1. The illustrations above all use SNLI. It is worth experimenting with MultiNLI as well. It has both __matched__ and __mismatched__ dev sets. It's also interesting to think about combining SNLI and MultiNLI, to get additional training instances, to push the models to generalize more, and to assess transfer learning hypotheses.
