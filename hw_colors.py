#!/usr/bin/env python
# coding: utf-8

# # Homework and bake-off: pragmatic color descriptions

# In[1]:


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


# ## Contents
# 
# 1. [Overview](#Overview)
# 1. [Set-up](#Set-up)
# 1. [All two-word examples as a dev corpus](#All-two-word-examples-as-a-dev-corpus)
# 1. [Dev dataset](#Dev-dataset)
# 1. [Random train–test split for development](#Random-train–test-split-for-development)
# 1. [Question 1: Improve the tokenizer [1 point]](#Question-1:-Improve-the-tokenizer-[1-point])
# 1. [Use the tokenizer](#Use-the-tokenizer)
# 1. [Question 2: Improve the color representations [1 point]](#Question-2:-Improve-the-color-representations-[1-point])
# 1. [Use the color representer](#Use-the-color-representer)
# 1. [Initial model](#Initial-model)
# 1. [Question 3: GloVe embeddings [1 points]](#Question-3:-GloVe-embeddings-[1-points])
# 1. [Try the GloVe representations](#Try-the-GloVe-representations)
# 1. [Question 4: Color context [3 points]](#Question-4:-Color-context-[3-points])
# 1. [Your original system [3 points]](#Your-original-system-[3-points])
# 1. [Bakeoff [1 point]](#Bakeoff-[1-point])

# ## Overview
# 
# This homework and associated bake-off are oriented toward building an effective system for generating color descriptions that are pragmatic in the sense that they would help a reader/listener figure out which color was being referred to in a shared context consisting of a target color (whose identity is known only to the describer/speaker) and a set of distractors.
# 
# The notebook [colors_overview.ipynb](colors_overview.ipynb) should be studied before work on this homework begins. That notebook provides backgroud on the task, the dataset, and the modeling code that you will be using and adapting.
# 
# The homework questions are more open-ended than previous ones have been. Rather than asking you to implement pre-defined functionality, they ask you to try to improve baseline components of the full system in ways that you find to be effective. As usual, this culiminates in a prompt asking you to develop a novel system for entry into the bake-off. In this case, though, the work you do for the homework will likely be directly incorporated into that system.

# ## Set-up

# See [colors_overview.ipynb](colors_overview.ipynb) for set-up in instructions and other background details.

# In[2]:


from colors import ColorsCorpusReader
import os
from sklearn.model_selection import train_test_split
from torch_color_describer import (
    ContextualColorDescriber, create_example_dataset)
import utils
from utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL


# In[3]:


utils.fix_random_seeds()


# In[4]:


COLORS_SRC_FILENAME = os.path.join(
    "data", "colors", "filteredCorpus.csv")


# ## All two-word examples as a dev corpus
# 
# So that you don't have to sit through excessively long training runs during development, I suggest working with the two-word-only subset of the corpus until you enter into the late stages of system testing.

# In[5]:


dev_corpus = ColorsCorpusReader(
    COLORS_SRC_FILENAME, 
    word_count=2, 
    normalize_colors=True)


# In[6]:


dev_examples = list(dev_corpus.read())


# This subset has about one-third the examples of the full corpus:

# In[7]:


len(dev_examples)


# We __should__ worry that it's not a fully representative sample. Most of the descriptions in the full corpus are shorter, and a large proportion are longer. So this dataset is mainly for debugging, development, and general hill-climbing. All findings should be validated on the full dataset at some point.

# ## Dev dataset
# 
# The first step is to extract the raw color and raw texts from the corpus:

# In[8]:


dev_rawcols, dev_texts = zip(*[[ex.colors, ex.contents] for ex in dev_examples])


# The raw color representations are suitable inputs to a model, but the texts are just strings, so they can't really be processed as-is. Question 1 asks you to do some tokenizing!

# ## Random train–test split for development
# 
# For the sake of development runs, we create a random train–test split:

# In[9]:


dev_rawcols_train, dev_rawcols_test, dev_texts_train, dev_texts_test =     train_test_split(dev_rawcols, dev_texts)


# In[10]:


print(len(dev_texts_train))
print(len(dev_texts_test))


# ## Question 1: Improve the tokenizer [1 point]
# 
# This is the first required question – the first required modification to the default pipeline.
# 
# The function `tokenize_example` simply splits its string on whitespace and adds the required start and end symbols:

# In[11]:


from transformers import BertTokenizer

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[12]:


def tokenize_example(s):
    
    # Improve me!
    
    # Tokenize our sentence with the BERT tokenizer.
    tokenized_s = tokenizer.tokenize(s)
    
    return [START_SYMBOL] + tokenized_s + [END_SYMBOL]


# In[13]:


print(dev_texts_train[376])
print(tokenize_example(dev_texts_train[376]))


# __Your task__: Modify `tokenize_example` so that it does something more sophisticated with the input text. 
# 
# __Notes__:
# 
# * There are useful ideas for this in [Monroe et al. 2017](https://transacl.org/ojs/index.php/tacl/article/view/1142)
# * There is no requirement that you do word-level tokenization. Sub-word and multi-word are options.
# * This question can interact with the size of your vocabulary (see just below), and in turn with decisions about how to use `UNK_SYMBOL`.
# 
# __Important__: don't forget to add the start and end symbols, else the resulting models will definitely be terrible!

# ## Use the tokenizer

# Once the tokenizer is working, run the following cell to tokenize your inputs:

# In[14]:


dev_seqs_train = [tokenize_example(s) for s in dev_texts_train]

dev_seqs_test = [tokenize_example(s) for s in dev_texts_test]


# We use only the train set to derive a vocabulary for the model:

# In[15]:


dev_vocab = sorted({w for toks in dev_seqs_train for w in toks}) + [UNK_SYMBOL]


# It's important that the `UNK_SYMBOL` is included somewhere in this list. Test examples with word not seen in training will be mapped to `UNK_SYMBOL`. If you model's vocab is the same as your train vocab, then `UNK_SYMBOL` will never be encountered during training, so it will be a random vector at test time.

# In[16]:


len(dev_vocab)


# ## Question 2: Improve the color representations [1 point]
# 
# This is the second required pipeline improvement for the assignment. 
# 
# The following functions do nothing at all to the raw input colors we get from the corpus. 

# In[17]:


import numpy as np
from skimage import color as clr

def represent_color_context(colors):
    
    # Improve me!
    
    return [represent_color(color) for color in colors]


def represent_color(color):
    # Improve me!
    
    color_hsv = np.array(color)  # color_hsv: (3,) . lets modify it to be able to use hsv2rgb
    color_hsv = color_hsv.reshape(1, 1, -1)  # color_hsv: (1,1,3)
    color_rgb = clr.hsv2rgb(color_hsv)       # color_rgb: (1,1,3)
    
    color_hsv_rgb = np.concatenate((color_hsv, color_rgb), axis=None)  # (6,)
    return color_hsv_rgb


# In[18]:


represent_color_context(dev_rawcols_train[0])


# __Your task__: Modify `represent_color_context` and/or `represent_color` to represent colors in a new way.
#     
# __Notes__:
# 
# * The Fourier-transform method of [Monroe et al. 2017](https://transacl.org/ojs/index.php/tacl/article/view/1142) is a proven choice.
# * You are not required to keep `represent_color`. This might be unnatural if you want to perform an operation on each color trio all at once.
# * For that matter, if you want to process all of the color contexts in the entire data set all at once, that is fine too, as long as you can also perform the operation at test time with an unknown number of examples being tested.

# ## Use the color representer

# The following cell just runs your `represent_color_context` on the train and test sets:

# In[19]:


dev_cols_train = [represent_color_context(colors) for colors in dev_rawcols_train]

dev_cols_test = [represent_color_context(colors) for colors in dev_rawcols_test]


# At this point, our preprocessing steps are complete, and we can fit a first model.

# ## Initial model
# 
# The first model is configured right now to be a small model run for just a few iterations. It should be enough to get traction, but it's unlikely to be a great model. You are free to modify this configuration if you wish; it is here just for demonstration and testing:

# In[20]:


dev_mod = ContextualColorDescriber(
    dev_vocab, 
    embed_dim=10, 
    hidden_dim=10, 
    max_iter=5, 
    batch_size=128)


# In[21]:


_ = dev_mod.fit(dev_cols_train, dev_seqs_train)


# As discussed in [colors_overview.ipynb](colors_overview.ipynb), our primary metric is `listener_accuracy`:

# In[22]:


dev_mod.listener_accuracy(dev_cols_test, dev_seqs_test)


# We can also see the model's predicted sequences given color context inputs:

# In[23]:


dev_mod.predict(dev_cols_test[:1])


# In[24]:


dev_seqs_test[:1]


# ## Question 3: GloVe embeddings [1 points]
# 
# The above model uses a random initial embedding, as configured by the decoder used by `ContextualColorDescriber`. This homework question asks you to consider using GloVe inputs. 
# 
# __Your task__: Complete `create_glove_embedding` so that it creates a GloVe embedding based on your model vocabulary. This isn't mean to be analytically challenging, but rather just to create a basis for you to try out other kinds of rich initialization.

# In[25]:


GLOVE_HOME = os.path.join('data', 'glove.6B')


# In[26]:


def create_glove_embedding(vocab, glove_base_filename='glove.6B.50d.txt'):
    
    # Use `utils.glove2dict` to read in the GloVe file:    
    ##### YOUR CODE HERE
    glove2dict = utils.glove2dict(os.path.join(GLOVE_HOME, glove_base_filename))

    
    # Use `utils.create_pretrained_embedding` to create the embedding.
    # This function will, by default, ensure that START_TOKEN, 
    # END_TOKEN, and UNK_TOKEN are included in the embedding.
    ##### YOUR CODE HERE
    embedding, vocab = utils.create_pretrained_embedding(glove2dict, vocab)


    
    # Be sure to return the embedding you create as well as the
    # vocabulary returned by `utils.create_pretrained_embedding`,
    # which is likely to have been modified from the input `vocab`.
    
    ##### YOUR CODE HERE
    return embedding, vocab



# ## Try the GloVe representations

# Let's see if GloVe helped for our development data:

# In[27]:


dev_glove_embedding, dev_glove_vocab = create_glove_embedding(dev_vocab)


# The above might dramatically change your vocabulary, depending on how many items from your vocab are in the Glove space:

# In[28]:


len(dev_vocab)


# In[29]:


len(dev_glove_vocab)


# In[30]:


dev_mod_glove = ContextualColorDescriber(
    dev_glove_vocab, 
    embedding=dev_glove_embedding,
    hidden_dim=10, 
    max_iter=5, 
    batch_size=128)


# In[31]:


_ = dev_mod_glove.fit(dev_cols_train, dev_seqs_train)


# In[32]:


dev_mod_glove.listener_accuracy(dev_cols_test, dev_seqs_test)


# You probably saw a small boost, assuming your tokeization scheme leads to good overlap with the GloVe vocabulary. The input representations are larger than in our previous model (at least as I configured things), so we would need to do more runs with higher `max_iter` values to see whether this is worthwhile overall.

# ## Question 4: Color context [3 points]
# 
# The final required homework question is the most challenging, but it should set you up to think in much more flexible ways about the underlying model we're using.
# 
# The question asks you to modify various model components in `torch_color_describer.py`. The section called [Modifying the core model](colors_overview.ipynb#Modifying-the-core-model) from the core unit notebook provides a number of examples illustrating the basic techniques, so you might review that material if you get stuck here.
# 
# __Your task__: [Monroe et al. 2017](https://transacl.org/ojs/index.php/tacl/article/view/1142) append the target color (the final one in the context) to each input token that gets processed by the decoder. The question asks you to subclass the `Decoder` and `EncoderDecoder` from `torch_color_describer.py` so that you can build models that do this.

# __Step 1__: Modify the `Decoder` so that the input vector to the model at each timestep is not just a token representaton `x` but the concatenation of `x` with the representation of the target color.
# 
# __Notes__:
# 
# * You might notice at this point that the original `Decoder.forward` method has an optional keyword argument `target_colors` that is passed to `Decoder.get_embeddings`. Because this is already in place, all you have to do is modify the `get_embeddings` method to use this argument.
# 
# * The change affects the configuration of `self.rnn`, so you need to subclass the `__init__` method as well, so that its `input_size` argument accomodates the embedding as well as the color representations.
# 
# * You can do the relevant operations efficiently in pure PyTorch using `repeat_interleave` and `cat`, but the important thing is to get a working implementation – you can always optimize the code later if the ideas prove useful to you. 
# 
# Here's skeleton code for you to flesh out:

# In[33]:


import torch
import torch.nn as nn
from torch_color_describer import Decoder

class ColorContextDecoder(Decoder):    
    def __init__(self, color_dim, *args, **kwargs):
        self.color_dim = color_dim
        super().__init__(*args, **kwargs)
        
        # Fix the `self.rnn` attribute:
        ##### YOUR CODE HERE
        self.rnn = nn.GRU(
            input_size=self.embed_dim + self.color_dim,
            hidden_size=self.hidden_dim,
            batch_first=True)


        

    def get_embeddings(self, word_seqs, target_colors=None):  
        """You can assume that `target_colors` is a tensor of shape 
        (m, n), where m is the length of the batch (same as 
        `word_seqs.shape[0]`) and n is the dimensionality of the 
        color representations the model is using. The goal is
        to attached each color vector i to each of the tokens in
        the ith sequence of (the embedded version of) `word_seqs`.
        
        """        
        ##### YOUR CODE HERE
        emb = self.embedding(word_seqs)  # shape : (m, k, embed_dim)
        m = emb.shape[0]
        k = emb.shape[1]

        p = target_colors.unsqueeze(1)  # add additional dimension.  p : (m, 1, color_dim)
        p = p.repeat_interleave(k, 1)  # repeat the *elements* of tensor along dimension 1.  p : (m, k, color_dim)
        emb2 = torch.cat((emb, p), 2)  # torch.cat  emb2 : (m, k, embed_dim + color_dim)
        return emb2


# __Step 2__: Modify the `EncoderDecoder`. For this, you just need to make a small change to the `forward` method: extract the target colors from `color_seqs` and feed them to the decoder.

# In[34]:


from torch_color_describer import EncoderDecoder

class ColorizedEncoderDecoder(EncoderDecoder):
    
    def forward(self, 
            color_seqs, 
            word_seqs, 
            seq_lengths=None, 
            hidden=None, 
            targets=None):
        if hidden is None:
            hidden = self.encoder(color_seqs)
            
        # Extract the target colors from `color_seqs` and 
        # feed them to the decoder, which already has a
        # `target_colors` keyword.        
        ##### YOUR CODE HERE
        output, hidden = self.decoder(
            word_seqs,
            seq_lengths=seq_lengths,
            hidden=hidden,
            target_colors=color_seqs[:,2,:])

        
        return output, hidden, targets


# __Step 3__: Finally, as in the examples in [Modifying the core model](colors_overview.ipynb#Modifying-the-core-model), you need to modify the `build_graph` method of `ContextualColorDescriber` so that it uses your new `ColorContextDecoder` and `ColorizedEncoderDecoder`. Here's starter code:

# In[35]:


from torch_color_describer import Encoder

class ColorizedInputDescriber(ContextualColorDescriber):
        
    def build_graph(self):
        
        # We didn't modify the encoder, so this is
        # just copied over from the original:
        encoder = Encoder(
            color_dim=self.color_dim,
            hidden_dim=self.hidden_dim)

        # Use your `ColorContextDecoder`, making sure
        # to pass in all the keyword arguments coming
        # from `ColorizedInputDescriber`:
        
        ##### YOUR CODE HERE
        decoder = ColorContextDecoder(
            self.color_dim,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            embedding=self.embedding,
            hidden_dim=self.hidden_dim)

        
        # Return a `ColorizedEncoderDecoder` that uses
        # your encoder and decoder:
        
        ##### YOUR CODE HERE
        return ColorizedEncoderDecoder(encoder, decoder)


# That's it! Since these modifications are pretty intricate, you might want to use [a toy dataset](colors_overview.ipynb#Toy-problems-for-development-work) to debug it:

# In[36]:


toy_color_seqs, toy_word_seqs, toy_vocab = create_example_dataset(
    group_size=50, vec_dim=2)


# In[37]:


toy_color_seqs_train, toy_color_seqs_test, toy_word_seqs_train, toy_word_seqs_test =     train_test_split(toy_color_seqs, toy_word_seqs)


# In[38]:


toy_mod = ColorizedInputDescriber(
    toy_vocab, 
    embed_dim=10, 
    hidden_dim=10, 
    max_iter=100, 
    batch_size=128)


# In[39]:


_ = toy_mod.fit(toy_color_seqs_train, toy_word_seqs_train)


# In[40]:


toy_mod.listener_accuracy(toy_color_seqs_test, toy_word_seqs_test)


# If that worked, then you can now try this model on SCC problems!

# ## Your original system [3 points]

# There are many options for your original system, which consists of the full pipeline – all preprocessing and modeling steps. You are free to use any model you like, as long as you subclass `ContextualColorDescriber` in a way that allows its `listener_accuracy` method to behave in the expected way.
# 
# So that we can evaluate models in a uniform way for the bake-off, we ask that you modify the function `my_original_system` below so that it accepts a trained instance of your model and does any preprocessing steps required by your model.
# 
# If we seek to reproduce your results, we will rerun this entire notebook. Thus, it is fine if your `my_original_system` makes use of functions you wrote or modified above this cell.

# In[41]:


def my_original_system(trained_model, color_seqs_test, texts_test): 
    """Feel free to modify this code to accommodate the needs of
    your system. Just keep in mind that it will get raw corpus
    examples as inputs for the bake-off.
    
    """    
    # `word_seqs_test` is a list of strings, so tokenize each of
    # its elements:    
    tok_seqs = [tokenize_example(s) for s in texts_test]
    
    col_seqs = [represent_color_context(colors) 
                for colors in color_seqs_test]

    # Return the `listener_accuracy` for your model:
    return trained_model.listener_accuracy(col_seqs, tok_seqs)


# If `my_original_system` works on test sets you create from the corpus distribution, then it will works for the bake-off, so consider checking that. For example, this would check that `dev_mod` above passes muster:

# In the cell below, please provide a brief technical description of your original system, so that the teaching team can gain an understanding of what it does. This will help us to understand your code and analyze all the submissions to identify patterns and strategies.

# In[42]:


# Enter your system description in this cell.

# Use of ColorizedInputDescriber with glove embeddings, based on BERT word tokenization
# hidden_dim 50 with larger number of iterations (100)

# My peak score was: 0.78

# Data prep    
corpus = ColorsCorpusReader(
    COLORS_SRC_FILENAME, 
    word_count=2, 
    normalize_colors=True)

# read data in, and get format we want.
examples = list(corpus.read())
rawcols, rawtexts = zip(*[[ex.colors, ex.contents] for ex in examples])
print("total # of examples:", len(rawtexts))

# train-test (for development only, not bakeoff)
rawcols_train, rawcols_test, rawtexts_train, rawtexts_test = train_test_split(rawcols, rawtexts)
print("train: {0}, test:{1}".format(len(rawtexts_train), len(rawtexts_test)))

# Color featurization (train split only)
fcols_train = [represent_color_context(colors) for colors in rawcols_train]

# Vocab featurization (train split only)
fseqs_train = [tokenize_example(s) for s in rawtexts_train]
vocab = sorted({w for toks in fseqs_train for w in toks}) + [UNK_SYMBOL]

glove_embedding, glove_vocab = create_glove_embedding(vocab)

# Train
model = ColorizedInputDescriber(
    glove_vocab, 
    embedding=glove_embedding, 
    hidden_dim=50, 
    max_iter=100, 
    batch_size=128)

_ = model.fit(fcols_train, fseqs_train)

# Eval metric
print("metric: ", my_original_system(model, rawcols_test, rawtexts_test))

# Please do not remove this comment.


# ## Bakeoff [1 point]

# For the bake-off, we will release a test set. The announcement will go out on the discussion forum. You will evaluate your custom model from the previous question on these new datasets using your `my_original_system` function. Rules:
# 
# 1. Only one evaluation is permitted.
# 1. No additional system tuning is permitted once the bake-off has started.
# 
# The cells below this one constitute your bake-off entry.
# 
# People who enter will receive the additional homework point, and people whose systems achieve the top score will receive an additional 0.5 points. We will test the top-performing systems ourselves, and only systems for which we can reproduce the reported results will win the extra 0.5 points.
# 
# Late entries will be accepted, but they cannot earn the extra 0.5 points. Similarly, you cannot win the bake-off unless your homework is submitted on time.
# 
# The announcement will include the details on where to submit your entry.

# In[43]:


# Enter your bake-off assessment code in this cell. 
# Please do not remove this comment.


# In[44]:


# On an otherwise blank line in this cell, please enter
# your listener_accuracy score as reported by the code
# above. Please enter only a number between 0 and 1 inclusive. 
# Please do not remove this comment.

