#!/usr/bin/env python
# coding: utf-8

# # Bringing contextual word representations into your models

# In[1]:


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


# ## Contents
# 
# 1. [Overview](#Overview)
# 1. [General set-up](#General-set-up)
# 1. [Hugging Face BERT interfaces](#Hugging-Face-BERT-interfaces)
#   1. [Hugging Face BERT set-up](#Hugging-Face-BERT-set-up)
#   1. [Hugging Face BERT basics](#Hugging-Face-BERT-basics)
#   1. [BERT featurization with Hugging Face](#BERT-featurization-with-Hugging-Face)
#     1. [Simple feed-forward experiment](#Simple-feed-forward-experiment)
#     1. [A feed-forward experiment with the sst module](#A-feed-forward-experiment-with-the-sst-module)
#     1. [An RNN experiment with the sst module](#An-RNN-experiment-with-the-sst-module)
#   1. [BERT fine-tuning with Hugging Face](#BERT-fine-tuning-with-Hugging-Face)
# 1. [Using ELMo](#Using-ELMo)
#   1. [ELMO Allen NLP set-up](#ELMO-Allen-NLP-set-up)
#   1. [ELMo featurization](#ELMo-featurization)
#     1. [ELMo featurization for an RNN](#ELMo-featurization-for-an-RNN)
#     1. [Using the SST experiment framework with ELMo](#Using-the-SST-experiment-framework-with-ELMo)
#   1. [ELMo fine-tuning](#ELMo-fine-tuning)

# ## Overview
# 
# This notebook provides a basic introduction to using pre-trained [BERT](https://github.com/google-research/bert) and [ELMo](https://allennlp.org/elmo) representations. It is meant as a practical companion to our lecture on contextual word representations. The goal of this notebook is just to help you use these representations in your own work. The BERT and ELMo teams have done amazing work to make these resources available to the community. Many projects can benefit from them, so it is probably worth your time to experiment.
# 
# This notebook should be considered an experimental extension to the regular course materials. It has some special requirements – libraries and data files – that are not part of the core requirements for this repository. All these tools are very new and being updated frequently, so you might need to do some fiddling to get all of this to work. As I said, though, it's probably worth the effort!
# 
# A number of the experiments in this notebook are resource intensive. I've included timing information for the expensive steps, to give you a sense for how long things are likely to take. I ran this notebook on a 2015 iMac with a 4 GHz Intel Core i7 CPU (no GPU involved) and 32GB of memory, and none of the steps seemed to strain the system. If you run this notebook on a GPU that PyTorch can work with, everything will be much faster.

# ## General set-up
# 
# The following are requirements that you'll already have met if you've been working in this repository. As you can see, we'll use the [Stanford Sentiment Treebank](sst_01_overview.ipynb) for illustrations, and we'll try out a few different deep learning models.

# In[2]:


import os
import sst
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from torch_rnn_classifier import TorchRNNClassifier, TorchRNNClassifierModel
from torch_rnn_classifier import TorchRNNClassifier
from sklearn.metrics import classification_report
import utils


# In[3]:


# Set all the random seeds for reproducibility. Only the
# system and torch seeds are relevant for this notebook.

utils.fix_random_seeds()


# In[4]:


SST_HOME = os.path.join("data", "trees")


# ## Hugging Face BERT interfaces

# ### Hugging Face BERT set-up
# 
# To install this library, run
# 
# ```pip install transformers```
# 
# I've tested this code with versions 2.4 and 2.5 of `transformers`. Try to get 2.5. It requires `pip >= 20` and, I think, a version of [Rust](https://www.rust-lang.org) at least as high as 1.21.1.

# In[5]:


import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


# The `transformers` library does a lot of logging. To avoid ending up with a cluttered notebook, I am changing the logging level. You might want to skip this as you scale up to building production systems, since the logging is very good – it gives you a lot of insights into what the models and code are doing.

# In[6]:


import logging
logger = logging.getLogger()
logger.level = logging.ERROR


# ### Hugging Face BERT basics
# 
# To start, let's get a feel for the basic API that `transformers` provides. The first step is specifying the pretrained parameters we'll be using:

# In[7]:


hf_weights_name = 'bert-base-cased'


# There are lots other options for pretrained weights. See [this section of the project README.md](https://github.com/huggingface/transformers#quick-tour) for a good overview and code that documents how these weights align with different Transformer model classes.

# Next, we specify a tokenizer and a model that match both each other and our choice of pretrained weights:

# In[8]:


hf_tokenizer = BertTokenizer.from_pretrained(hf_weights_name)


# In[9]:


hf_model = BertModel.from_pretrained(hf_weights_name)


# It's illuminating to see what the tokenizer does to example texts:

# In[10]:


hf_example_texts = [
    "Encode sentence 1. [SEP] And sentence 2!",
    "Bert knows Snuffleupagus"]


# The `encode` method maps individual strings to indices into the underlying embedding used by the model:

# In[11]:


ex0_ids = hf_tokenizer.encode(hf_example_texts[0], add_special_tokens=True)

ex0_ids


# We can get a better feel for what these representations are like by mapping the indices back to "words":

# In[12]:


hf_tokenizer.convert_ids_to_tokens(ex0_ids)


# For modeling, we will often need to pad (and perhaps truncate) token lists so that we can work with fixed-dimensional tensors: The `batch_encode_plus` has a lot of options for doing this:

# In[13]:


hf_example_ids = hf_tokenizer.batch_encode_plus(
    hf_example_texts, 
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True)


# In[14]:


hf_example_ids.keys()


# The `token_type_ids` is used for multi-text inputs like NLI. The `'input_ids'` field gives the indices for each of the two examples:

# In[15]:


hf_example_ids['input_ids']


# For fine-tuning, we want to avoid attending to padded tokens. The `'attention_mask'` captures the needed mask, which we'll be able to feed directly to the pretrained BERT model:

# In[16]:


hf_example_ids['attention_mask']


# Finally, we can run these indices and masks through the pretrained model:

# In[17]:


X_hf_example = torch.tensor(hf_example_ids['input_ids'])
X_hf_example_mask = torch.tensor(hf_example_ids['attention_mask'])

with torch.no_grad():    
    hf_final_hidden_states, cls_output = hf_model(
        X_hf_example, attention_mask=X_hf_example_mask)


# BERT representations are pretty large – this shows 2 examples, with the second padded to the length of the larger one in the batch (12). The individual representations have dimensionality 768.

# In[18]:


hf_final_hidden_states.shape


# Those are all the essential ingredients for working with these parameters in Hugging Face. Of course, the library has a lot of other functionality, but the above suffices to featurize and to fine tune.

# ### BERT featurization with Hugging Face
# 
# To start, we'll use the Hugging Face interfaces just to featurize examples to create inputs to a separate model. In this setting, the BERT parameters are frozen. The heart of this approach is the following featurizer, which flattens an SST tree into a string, tokenizes it, and calculates its hidden representations:

# In[19]:


def hugging_face_bert_phi(tree):
    s = " ".join(tree.leaves())
    input_ids = hf_tokenizer.encode(s, add_special_tokens=True)
    X = torch.tensor([input_ids])
    with torch.no_grad():
        final_hidden_states, cls_output = hf_model(X)
        return final_hidden_states.squeeze(0).numpy() 


# #### Simple feed-forward experiment
# 
# For a simple feed-forward experiment, we can get the representation of the `[CLS]` tokens and use them as the inputs to a shallow neural network:

# In[20]:


def hugging_face_bert_classifier_phi(tree):
    reps = hugging_face_bert_phi(tree)
    #return reps.mean(axis=0)  # Another good, easy option.
    return reps[0]


# This is very much like what we [summed the GloVe representations of these examples](sst_03_neural_networks.ipynb#Distributed-representations-as-features), but now the individual word representations are different depending on the context in which they appear.

# Next we read in the SST train and dev portions as a lists of `(tree, label)` pairs:

# In[21]:


hf_train = list(sst.train_reader(SST_HOME, class_func=sst.ternary_class_func))

hf_dev = list(sst.dev_reader(SST_HOME, class_func=sst.ternary_class_func))


# Split the input/output pairs out into separate lists:

# In[22]:


X_hf_tree_train, y_hf_train = zip(*hf_train)

X_hf_tree_dev, y_hf_dev = zip(*hf_dev)


# In the next step, we featurize all of the examples. These steps are likely to be the slowest in these experiments:

# In[23]:


get_ipython().run_line_magic('time', 'X_hf_train = [hugging_face_bert_classifier_phi(tree) for tree in X_hf_tree_train]')


# In[24]:


get_ipython().run_line_magic('time', 'X_hf_dev = [hugging_face_bert_classifier_phi(tree) for tree in X_hf_tree_dev]')


# Now that all the examples are featurized, we can fit a model and evaluate it:

# In[25]:


hf_mod = TorchShallowNeuralClassifier(max_iter=100, hidden_dim=300)


# In[26]:


get_ipython().run_line_magic('time', '_ = hf_mod.fit(X_hf_train, y_hf_train)')


# In[27]:


hf_preds = hf_mod.predict(X_hf_dev)


# In[28]:


print(classification_report(y_hf_dev, hf_preds, digits=3))


# #### A feed-forward experiment with the sst module
# 
# It is straightforward to conduct experiments like the above using `sst.experiment`, which will enable you to do a wider range of experiments without writing or copy-pasting a lot of code. 

# In[29]:


def fit_hf_shallow_network(X, y):
    mod = TorchShallowNeuralClassifier(
        max_iter=100, hidden_dim=300)
    mod.fit(X, y)
    return mod


# In[30]:


get_ipython().run_cell_magic('time', '', '_ = sst.experiment(\n    SST_HOME,\n    hugging_face_bert_classifier_phi,\n    fit_hf_shallow_network,\n    train_reader=sst.train_reader, \n    assess_reader=sst.dev_reader, \n    class_func=sst.ternary_class_func,\n    vectorize=False)  # Pass in the BERT hidden state directly!')


# #### An RNN experiment with the sst module
# 
# We can also use BERT representations as the input to an RNN. There is just one key change from how we used these models before:
# 
# * Previously, we would feed in lists of tokens, and they would be converted to indices into a fixed embedding space. This presumes that all words have the same representation no matter what their context is. 
# 
# * With BERT, we skip the embedding entirely and just feed in lists of BERT vectors, which means that the same word can be represented in different ways.
# 
# `TorchRNNClassifier` supports this via `use_embedding=False`. In turn, you needn't supply a vocabulary:

# In[31]:


def fit_hf_rnn(X, y):
    mod = TorchRNNClassifier(
        vocab=[],
        max_iter=50, 
        hidden_dim=50,
        use_embedding=False)  # Pass in the BERT hidden states directly!
    mod.fit(X, y)
    return mod


# In[32]:


get_ipython().run_cell_magic('time', '', '_ = sst.experiment(\n    SST_HOME,\n    hugging_face_bert_phi,\n    fit_hf_rnn,\n    train_reader=sst.train_reader, \n    assess_reader=sst.dev_reader, \n    class_func=sst.ternary_class_func,\n    vectorize=False)  # Pass in the BERT hidden states directly!')


# ### BERT fine-tuning with Hugging Face
# 
# The above experiments are quite successful – BERT gives us a reliable boost compared to other methods we've explored for the SST task. However, we might expect to do even better if we fine-tune the BERT parameters as part of fitting our SST classifier. To do that, we need to incorporate the Hugging Face BERT model into our classifier. This too is quite straightforward.
# 
# The most important step is to create an `nn.Module` subclass that has, for its parameters, both the BERT model and parameters for our own classifier:

# In[33]:


class HfBertClassifierModel(nn.Module):
    def __init__(self, n_classes, weights_name='bert-base-cased'):
        super().__init__()
        self.n_classes = n_classes
        self.weights_name = weights_name
        self.bert = BertModel.from_pretrained(self.weights_name)
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim
        # The only new parameters -- the classifier layer:
        self.W = nn.Linear(self.hidden_dim, self.n_classes)
        
    def forward(self, X):
        """Here, `X` is an np.array in which each element is a pair 
        consisting of an index into the BERT embedding and a 1 or 0
        indicating whether the token is masked. The `fit` method will 
        train all these parameters against a softmax objective.
        
        """
        indices = X[: , 0, : ]
        # Type conversion, since the base class insists on
        # casting this as a FloatTensor, but we ned Long
        # for `bert`.
        indices = indices.long()
        mask = X[: , 1, : ]      
        (final_hidden_states, cls_output) = self.bert(
            indices, attention_mask=mask)       
        return self.W(cls_output)


# For the training and prediction interface, we can somewhat opportunistically subclass `TorchShallowNeuralClassifier` so that we don't have to write any of our own data-handling, training, or prediction code:

# In[34]:


class HfBertClassifier(TorchShallowNeuralClassifier):
    def __init__(self, weights_name, *args, **kwargs):
        self.weights_name = weights_name
        self.tokenizer = BertTokenizer.from_pretrained(self.weights_name)
        super().__init__(*args, **kwargs)
        
    def define_graph(self):
        """This method is used by `fit`. We override it here to use our
        new BERT-based graph.
        
        """
        bert = HfBertClassifierModel(
            self.n_classes_, weights_name=self.weights_name)
        bert.train()
        return bert
    
    def encode(self, X, max_length=None):
        """The `X` is a list of strings. We use the model's tokenizer
        to get the indices and mask information.
        
        Returns
        -------
        list of [index, mask] pairs, where index is an int and mask
        is 0 or 1.
        
        """
        data = self.tokenizer.batch_encode_plus(
            X, 
            max_length=max_length,
            add_special_tokens=True, 
            pad_to_max_length=True,
            return_attention_mask=True)
        indices = data['input_ids']
        mask = data['attention_mask']
        return [[i, m] for i, m in zip(indices, mask)]


# Here's a self-contained illustration, starting from the raw SST data:

# In[35]:


hf_train = list(sst.train_reader(SST_HOME, class_func=sst.ternary_class_func))

hf_dev = list(sst.dev_reader(SST_HOME, class_func=sst.ternary_class_func))

X_hf_tree_train, y_hf_train = zip(*hf_train)

X_hf_tree_dev, y_hf_dev = zip(*hf_dev)


# Our model has some standard fine-tuning parameters:

# In[36]:


hf_fine_tune_mod = HfBertClassifier(
    'bert-base-cased', 
    batch_size=16, # Crucial; large batches will eat up all your memory!
    max_iter=4, 
    eta=0.00002)


# We'll use the `encode` method in `HfBertClassifier`. For generality, this is expecting string inputs, rather than trees, so we first flatten the trees:

# In[37]:


X_hf_str_train = [" ".join(tree.leaves()) for tree in X_hf_tree_train]

X_hf_str_dev = [" ".join(tree.leaves()) for tree in X_hf_tree_dev]


# Now we can encode them; this step packs together the indices and mask information:

# In[38]:


X_hf_indices_train = hf_fine_tune_mod.encode(X_hf_str_train)

X_hf_indices_dev = hf_fine_tune_mod.encode(X_hf_str_dev)


# Training this model is resource intensive. Be patient – it will be worth the wait! (This experiment takes about 10 minutes on a machine with an NVIDIA RTX 2080 Max-Q GPU.)

# In[39]:


get_ipython().run_line_magic('time', '_ = hf_fine_tune_mod.fit(X_hf_indices_train, y_hf_train)')


# Finally, some predictions on the dev set:

# In[40]:


hf_fine_tune_preds = hf_fine_tune_mod.predict(X_hf_indices_dev)


# In[41]:


print(classification_report(hf_fine_tune_preds, y_hf_dev, digits=3))


# The above is just one of the many possible ways to fine-tune BERT using our course modules or new modules you write. The crux of it is creating an `nn.Module` that combines the BERT parameters with your model's new parameters.

# ## Using ELMo

# ### ELMO Allen NLP set-up
# 
# There are a number of ways to use pre-trained ELMo models. We'll use the simplest of the AllenNLP interfaces. Run the following to install [AllenNLP](https://allennlp.org):
# 
# ```sh
# pip install allennlp
# ```
# I've tested this notebook with versions 0.8.0 and 0.9.0.
# 
# Mac users: If your installation fails, make sure your Xcode tools are up to date by running `xcode-select --install`.

# In[42]:


from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
import torch
import torch.nn as nn


# We'll use the following models, which will download from S3 to a local temp directory the first time you use them with `ElmoEmbedder` or `Elmo` as described below.

# In[43]:


elmo_file_path = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/"

options_file = elmo_file_path + "elmo_2x4096_512_2048cnn_2xhighway_options.json"

weights_file = elmo_file_path + "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


# For more models:
# 
# https://allennlp.org/elmo
# 
# For additional details:
# 
# https://github.com/allenai/allennlp/blob/master/allennlp/commands/elmo.py

# ### ELMo featurization
# 
# As we did with BERT, we'll first use ELMo just to featurize examples for a separate model. The `ElmoEmbedder` facilitates this:

# In[44]:


elmo_embedder = ElmoEmbedder(options_file, weights_file)


# Here's a copy of the SST in a useful format for this modeling:

# In[45]:


elmo_train = list(sst.train_reader(SST_HOME, class_func=sst.ternary_class_func))

elmo_dev = list(sst.dev_reader(SST_HOME, class_func=sst.ternary_class_func))

X_elmo_tree_train, y_elmo_train = zip(*elmo_train)

X_elmo_tree_dev, y_elmo_dev = zip(*elmo_dev)


# The ELMo interface requires tokenized input. I believe the tokenization scheme is the same as for the SST, so we can just use the leaves of those trees:

# In[46]:


X_elmo_toks_train = [tree.leaves() for tree in X_elmo_tree_train]

X_elmo_toks_dev = [tree.leaves() for tree in X_elmo_tree_dev]


# #### ELMo featurization for an RNN

# Here we create the representations for the train and dev sets; these steps are somewhat slow:

# In[47]:


get_ipython().run_line_magic('time', 'X_elmo_train_layers = list(elmo_embedder.embed_sentences(X_elmo_toks_train))')


# In[48]:


get_ipython().run_line_magic('time', 'X_elmo_dev_layers = list(elmo_embedder.embed_sentences(X_elmo_toks_dev))')


# Each member of `X_elmo_train_layers` has three dimensions:

# In[49]:


X_elmo_dev_layers[0].shape


# For each word (second dimension), there are three layers of length 1024. So ELMo representations are even larger than BERT ones!
# 
# There are many ways we could combine the layers available for each word. Here, I'll use just the top layer; see [section 3.2 of the ELMo paper](https://www.aclweb.org/anthology/N18-1202/) for additional ideas.

# In[50]:


def elmo_layer_reduce_top(elmo_vecs):
    return [ex[-1] for ex in elmo_vecs]


# In[51]:


X_elmo_train = elmo_layer_reduce_top(X_elmo_train_layers)


# Now we can fit an RNN as usual:

# In[52]:


elmo_rnn = TorchRNNClassifier(
    vocab=[],
    max_iter=50,
    use_embedding=False) # Pass in the ELMo hidden states directly!


# In[53]:


get_ipython().run_line_magic('time', '_ = elmo_rnn.fit(X_elmo_train, y_elmo_train)')


# Evaluation proceeds in the usual way:

# In[54]:


X_elmo_dev = elmo_layer_reduce_top(X_elmo_dev_layers)


# In[55]:


elmo_rnn_preds = elmo_rnn.predict(X_elmo_dev)


# In[56]:


print(classification_report(y_elmo_dev, elmo_rnn_preds, digits=3))


# #### Using the SST experiment framework with ELMo
# 
# To round things out, here's an example of how to use `sst.experiment` with ELMo, for more compact and maintainable experiment code:

# In[57]:


def elmo_sentence_phi(tree):
    vecs = elmo_embedder.embed_sentence(tree.leaves())
    return vecs[-1]


# In[58]:


def fit_elmo_rnn(X, y):
    mod = TorchRNNClassifier(
        vocab=[],
        max_iter=50,
        use_embedding=False)
    mod.fit(X, y)
    return mod


# This step re-encodes all of the examples, so it will take a while before the model starts training:

# In[59]:


get_ipython().run_cell_magic('time', '', '_ = sst.experiment(\n    SST_HOME,\n    elmo_sentence_phi,\n    fit_elmo_rnn,\n    train_reader=sst.train_reader, \n    assess_reader=sst.dev_reader, \n    class_func=sst.ternary_class_func,\n    vectorize=False)  # Pass in the ELMo hidden states directly!')


# ### ELMo fine-tuning
# 
# Fine-tuning ELMo proceeds in essentially the same way it did for BERT: we create an `nn.Module` that combines the parameters from ELMo with our task-specific parameters and then optimize everything on the new task. To illustrate, I'll define an RNN on top of the ELMo model using new subclasses of `TorchRNNClassifier` and `TorchRNNClassifierModel`.
# 
# To start, let's get a feel for the primary interface, and then we'll write the classes that will allow us to use these components systematically.
# 
# The interface to the ELMo parameters in this context is the class `Elmo`:

# In[60]:


elmo = Elmo(options_file, weights_file, num_output_representations=2)


# This model expects tokenized inputs:

# In[61]:


elmo_example_texts = [
    ["Encode", "sentence", "1", "."],
    ["ELMo", "knows" "Snuffleupagus"]]


# The ELMo model processes its tokens at the character-level, creating convolutional representations for the words from various character n-gram combinations:

# In[62]:


elmo_character_ids = batch_to_ids(elmo_example_texts)

# First word of the first example:
elmo_character_ids[0][0]


# `elmo` embeds these at the word-level:

# In[63]:


elmo_embeddings = elmo(elmo_character_ids)


# `elmo_embeddings` is a dict. The value of `'elmo_representations'` is a list of tensors corresponding to each layer of the model. In other words, each tensor in the list is a complete representation of the example. The final element of the list is the final representation layer.

# In[64]:


elmo_embeddings['elmo_representations']


# These are the representations we will be fine-tuning. There are many ways to combine use. In my simple illustration, I just take the top layer, as we did in the simpler featurization example above, but now keeping each word representation separate for use in the input to the task-specific RNN. Here is the `nn.Module`:

# In[65]:


class ElmoRNNClassifierModel(TorchRNNClassifierModel):
    def __init__(self, options_file, weights_file, hidden_dim, output_dim, bidirectional, device):     
        super().__init__(
            vocab_size=0,
            embed_dim=1024, # self.elmo.get_output_dim()
            use_embedding=False,
            embedding=None,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            bidirectional=bidirectional,
            device=device)
        self.options_file = options_file
        self.weights_file = weights_file
        self.elmo = Elmo(
            self.options_file, 
            self.weights_file, 
            num_output_representations=2, 
            dropout=0)
        
    def forward(self, X, seq_lengths):
        X = X.to(self.device, non_blocking=True)
        result = self.elmo(X)
        X = result['elmo_representations'][-1]
        state = self.rnn_forward(X, seq_lengths, self.rnn)
        logits = self.classifier_layer(state)
        return logits


# And here is the subclass of `TorchRNNClassifier` that lets us take advantage of all the optimization and prediction methods of that class:

# In[66]:


class ElmoRNNClassifier(TorchRNNClassifier):
    def __init__(self, options_file, weights_file, *args, **kwargs):
        self.options_file = options_file
        self.weights_file = weights_file
        vocab = []
        super().__init__(vocab, *args, use_embedding=False, embedding=None, **kwargs)
        
    def build_graph(self):
        """This method is used by `fit`. We override it here to use our
        new ELMo-based graph.
        
        """
        elmo = ElmoRNNClassifierModel(           
            options_file=self.options_file,
            weights_file=self.weights_file,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_classes_,
            bidirectional=self.bidirectional,
            device=self.device)        
        elmo.train()
        return elmo
    
    def _prepare_dataset(self, X):
        # Somewhat awkwardly get the lengths (padded tokens are all 0s).
        # Ideally, this function would first measure the lengths of the
        # examples in X directly and then call `batch_to_ids(X)`. 
        # However, the super class `fit` method is currently presupposing
        # too much about X to allow this. TODO: make that interface
        # more flexible.
        seq_lengths = [sum([1 for w in ex if w.sum() > 0]) for ex in X]
        return X, torch.tensor(seq_lengths)
    
    @staticmethod
    def encode(X):
        return batch_to_ids(X)


# And finally here is a self-contained illustration:
# 
# A copy of the dataset in the format we want:

# In[67]:


elmo_train = list(sst.train_reader(SST_HOME, class_func=sst.ternary_class_func))

elmo_dev = list(sst.dev_reader(SST_HOME, class_func=sst.ternary_class_func))

X_elmo_tree_train, y_elmo_train = zip(*elmo_train)

X_elmo_tree_dev, y_elmo_dev = zip(*elmo_dev)

X_elmo_toks_train = [tree.leaves() for tree in X_elmo_tree_train]

X_elmo_toks_dev = [tree.leaves() for tree in X_elmo_tree_dev]


# Our fine-tuning model with parameters that are inspired by those given for the SST task in [the paper's supplementary materials](https://www.aclweb.org/anthology/attachments/N18-1202.Notes.pdf):

# In[68]:


elmo_fine_tune_mod = ElmoRNNClassifier(
    options_file, 
    weights_file, 
    batch_size=16, 
    max_iter=10,  # More iters improves performance. How many did the ELMo team do?
    eta=0.0001,
    l2_strength=0.0001)


# Character-level encodings:

# In[69]:


X_elmo_train = elmo_fine_tune_mod.encode(X_elmo_toks_train)

X_elmo_dev = elmo_fine_tune_mod.encode(X_elmo_toks_dev)


# Train (This experiment takes about 11 minutes on a machine with an NVIDIA RTX 2080 Max-Q GPU):

# In[70]:


get_ipython().run_line_magic('time', '_ = elmo_fine_tune_mod.fit(X_elmo_train, y_elmo_train)')


# Assess:

# In[71]:


# When I was on a GPU machine, I had trouble predicting on the 
# full dev set all at once, so let's break it into small batches:
increment = 10
elmo_fine_tune_preds = []
for i in range(0, len(X_elmo_dev), increment):
    elmo_fine_tune_preds += elmo_fine_tune_mod.predict(X_elmo_dev[i: i+increment])


# In[72]:


print(classification_report(elmo_fine_tune_preds, y_elmo_dev, digits=3))

