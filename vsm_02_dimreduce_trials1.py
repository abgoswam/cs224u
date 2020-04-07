#!/usr/bin/env python
# coding: utf-8

# # Vector-space models: dimensionality reduction

# In[1]:


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


# ## Contents
# 
# 1. [Overview](#Overview)
# 1. [Set-up](#Set-up)
# 1. [Latent Semantic Analysis](#Latent-Semantic-Analysis)
#   1. [Overview of the LSA method](#Overview-of-the-LSA-method)
#   1. [Motivating example for LSA](#Motivating-example-for-LSA)
#   1. [Applying LSA to real VSMs](#Applying-LSA-to-real-VSMs)
#   1. [Other resources for matrix factorization](#Other-resources-for-matrix-factorization)
# 1. [GloVe](#GloVe)
#   1. [Overview of the GloVe method](#Overview-of-the-GloVe-method)
#   1. [GloVe implementation notes](#GloVe-implementation-notes)
#   1. [Applying GloVe to our motivating example](#Applying-GloVe-to-our-motivating-example)
#   1. [Testing the GloVe implementation](#Testing-the-GloVe-implementation)
#   1. [Applying GloVe to real VSMs](#Applying-GloVe-to-real-VSMs)
# 1. [Autoencoders](#Autoencoders)
#   1. [Overview of the autoencoder method](#Overview-of-the-autoencoder-method)
#   1. [Testing the autoencoder implementation](#Testing-the-autoencoder-implementation)
#   1. [Applying autoencoders to real VSMs](#Applying-autoencoders-to-real-VSMs)
# 1. [word2vec](#word2vec)
#   1. [Training data](#Training-data)
#   1. [Basic skip-gram](#Basic-skip-gram)
#   1. [Skip-gram with noise contrastive estimation ](#Skip-gram-with-noise-contrastive-estimation-)
#   1. [word2vec resources](#word2vec-resources)
# 1. [Other methods](#Other-methods)
# 1. [Exploratory exercises](#Exploratory-exercises)

# ## Overview
# 
# The matrix weighting schemes reviewed in the first notebook for this unit deliver solid results. However, they are not capable of capturing higher-order associations in the data. 
# 
# With dimensionality reduction, the goal is to eliminate correlations in the input VSM and capture such higher-order notions of co-occurrence, thereby improving the overall space.
# 
# As a motivating example, consider the adjectives _gnarly_ and _wicked_ used as slang positive adjectives.  Since both are positive, we expect them to be similar in a good VSM. However, at least stereotypically, _gnarly_ is Californian and _wicked_ is Bostonian. Thus, they are unlikely to occur often in the same texts, and so the methods we've reviewed so far will not be able to model their similarity. 
# 
# Dimensionality reduction techniques are often capable of capturing such semantic similarities (and have the added advantage of shrinking the size of our data structures).

# ## Set-up
# 
# * Make sure your environment meets all the requirements for [the cs224u repository](https://github.com/cgpotts/cs224u/). For help getting set-up, see [setup.ipynb](setup.ipynb).
# 
# * Make sure you've downloaded [the data distribution for this course](http://web.stanford.edu/class/cs224u/data/data.zip), unpacked it, and placed it in the current directory (or wherever you point `DATA_HOME` to below).

# In[2]:


from mittens import GloVe
import numpy as np
import os
import pandas as pd
import scipy.stats
from torch_autoencoder import TorchAutoencoder
import utils
import vsm


# In[3]:


# Set all the random seeds for reproducibility:

utils.fix_random_seeds()


# In[4]:


DATA_HOME = os.path.join('data', 'vsmdata')


# In[5]:


imdb5 = pd.read_csv(
    os.path.join(DATA_HOME, 'imdb_window5-scaled.csv.gz'), index_col=0)


# In[6]:


imdb20 = pd.read_csv(
    os.path.join(DATA_HOME, 'imdb_window20-flat.csv.gz'), index_col=0)


# In[7]:


giga5 = pd.read_csv(
    os.path.join(DATA_HOME, 'giga_window5-scaled.csv.gz'), index_col=0)


# In[8]:


giga20 = pd.read_csv(
    os.path.join(DATA_HOME, 'giga_window20-flat.csv.gz'), index_col=0)


# ## Latent Semantic Analysis
# 
# Latent Semantic Analysis (LSA) is a prominent dimensionality reduction technique. It is an application of __truncated singular value decomposition__ (SVD) and so uses only techniques from linear algebra (no machine learning needed).

# ### Overview of the LSA method
# 
# The central mathematical result is that, for any matrix of real numbers $X$ of dimension $m \times n$, there is a factorization of $X$ into matrices $T$, $S$, and $D$ such that
# 
# $$X_{m \times n} = T_{m \times m}S_{m\times m}D_{n \times m}^{\top}$$
# 
# The matrices $T$ and $D$ are  __orthonormal__ – their columns are length-normalized and orthogonal to one another (that is, they each have cosine distance of $1$ from each other). The singular-value matrix $S$ is a diagonal matrix arranged by size, so that the first dimension corresponds to the greatest source of variability in the data, followed by the second, and so on.
# 
# Of course, we don't want to factorize and rebuild the original matrix, as that wouldn't get us anywhere. The __truncation__ part means that we include only the top $k$ dimensions of $S$. Given our row-oriented perspective on these matrices, this means using
# 
# $$T[1{:}m, 1{:}k]S[1{:}k, 1{:}k]$$
# 
# which gives us a version of $T$ that includes only the top $k$ dimensions of variation. 
# 
# To build up intuitions, imagine that everyone on the Stanford campus is associated with a 3d point representing their position: $x$ is east–west, $y$ is north–south, and $z$ is zenith–nadir. Since the campus is spread out and has relatively few deep basements and tall buildings, the top two dimensions of variation will be $x$ and $y$, and the 2d truncated SVD of this space will leave $z$ out. This will, for example, capture the sense in which someone at the top of Hoover Tower is close to someone at its base.

# ### Motivating example for LSA
# 
# We can also return to our original motivating example of _wicked_ and _gnarly_. Here is a matrix reflecting those assumptions:

# In[9]:


gnarly_df = pd.DataFrame(
    np.array([
        [1,0,1,0,0,0],
        [0,1,0,1,0,0],
        [1,1,1,1,0,0],
        [0,0,0,0,1,1],
        [0,0,0,0,0,1]], dtype='float64'),
    index=['gnarly', 'wicked', 'awesome', 'lame', 'terrible'])

gnarly_df


# No column context includes both _gnarly_ and _wicked_ together so our count matrix places them far apart:

# In[10]:


vsm.neighbors('gnarly', gnarly_df)


# Reweighting doesn't help. For example, here is the attempt with Positive PMI:

# In[11]:


vsm.neighbors('gnarly', vsm.pmi(gnarly_df))


# However, both words tend to occur with _awesome_ and not with _lame_ or _terrible_, so there is an important sense in which they are similar. LSA to the rescue:

# In[12]:


gnarly_lsa_df = vsm.lsa(gnarly_df, k=2)


# In[13]:


vsm.neighbors('gnarly', gnarly_lsa_df)


# ### Applying LSA to real VSMs
# 
# Here's an example that begins to convey the effect that this can have empirically.
# 
# First, the original count matrix:

# In[14]:


vsm.neighbors('superb', imdb5).head()


# And then LSA with $k=100$:

# In[15]:


imdb5_svd = vsm.lsa(imdb5, k=100)


# In[16]:


vsm.neighbors('superb', imdb5_svd).head()


# A common pattern in the literature is to apply PMI first. The PMI values tend to give the count matrix a normal (Gaussian) distribution that better satisfies the assumptions underlying SVD:

# In[17]:


imdb5_pmi = vsm.pmi(imdb5, positive=False)


# In[18]:


imdb5_pmi_svd = vsm.lsa(imdb5_pmi, k=100)


# In[19]:


vsm.neighbors('superb', imdb5_pmi_svd).head()


# ### Other resources for matrix factorization
# 
# The [sklearn.decomposition](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition) module contains an implementation of LSA ([TruncatedSVD](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD)) that you might want to switch to for real experiments:
# 
# * The `sklearn` version is more flexible than the above in that it can operate on both dense matrices (Numpy arrays) and sparse matrices (from Scipy).
# 
# * The `sklearn` version will make it easy to try out other dimensionality reduction methods in your own code; [Principal Component Analysis (PCA)](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA) and [Non-Negative Matrix Factorization (NMF)](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF) are closely related methods that are worth a look.

# ## GloVe
# 
# ### Overview of the GloVe method
# 
# [Pennington et al. (2014)](http://www.aclweb.org/anthology/D/D14/D14-1162.pdf) introduce an objective function for semantic word representations. Roughly speaking, the objective is to learn vectors for words $w_{i}$ and $w_{j}$ such that their dot product is proportional to their probability of co-occurrence:
# 
# $$w_{i}^{\top}\widetilde{w}_{k} + b_{i} + \widetilde{b}_{k} = \log(X_{ik})$$
# 
# The paper is exceptionally good at motivating this objective from first principles. In their equation (6), they define 
# 
# $$w_{i}^{\top}\widetilde{w}_{k} = \log(P_{ik}) = \log(X_{ik}) - \log(X_{i})$$
# 
# If we allow that the rows and columns can be different, then we would do
# 
# $$w_{i}^{\top}\widetilde{w}_{k} = \log(P_{ik}) = \log(X_{ik}) - \log(X_{i} \cdot X_{*k})$$
# 
# where, as in the paper, $X_{i}$ is the sum of the values in row $i$, and $X_{*k}$ is the sum of the values in column $k$.
# 
# The rightmost expression is PMI by the equivalence $\log(\frac{x}{y}) = \log(x) - \log(y)$, and hence we can see GloVe as aiming to make the dot product of two learned vectors equal to the PMI!
# 
# The full model is a weighting of this objective:
# 
# $$\sum_{i, j=1}^{|V|} f\left(X_{ij}\right)
#   \left(w_i^\top \widetilde{w}_j + b_i + \widetilde{b}_j - \log X_{ij}\right)^2$$
# 
# where $V$ is the vocabulary and $f$ is a scaling factor designed to diminish the impact of very large co-occurrence counts:
# 
# $$f(x) 
# \begin{cases}
# (x/x_{\max})^{\alpha} & \textrm{if } x < x_{\max} \\
# 1 & \textrm{otherwise}
# \end{cases}$$
# 
# Typically, $\alpha$ is set to $0.75$ and $x_{\max}$ to $100$ (though it is worth assessing how many of your non-zero counts are above this; in dense word $\times$ word matrices, you could be flattening more than you want to).

# ### GloVe implementation notes
# 
# * The implementation in `vsm.glove` is the most stripped-down, bare-bones version of the GloVe method I could think of. As such, it is quite slow. 
# 
# * The required [mittens](https://github.com/roamanalytics/mittens) package includes a vectorized implementation that is much, much faster, so we'll mainly use that. 
# 
# * For really large jobs, [the official C implementation released by the GloVe team](http://nlp.stanford.edu/projects/glove/) is probably the best choice.

# ### Applying GloVe to our motivating example
# 
# GloVe should do well on our _gnarly/wicked_ evaluation, though you will see a lot variation due to the small size of this VSM:

# In[20]:


gnarly_glove = vsm.glove(gnarly_df, n=5, max_iter=1000)


# In[21]:


vsm.neighbors('gnarly', gnarly_glove)


# ### Testing the GloVe implementation
# 
# It is not easy analyze GloVe values derived from real data, but the following little simulation suggests that `vsm.glove` is working as advertised: it does seem to reliably deliver vectors whose dot products are proportional to the log co-occurrence probability:

# In[22]:


glove_test_count_df = pd.DataFrame(
    np.array([
        [10.0,  2.0,  3.0,  4.0],
        [ 2.0, 10.0,  4.0,  1.0],
        [ 3.0,  4.0, 10.0,  2.0],
        [ 4.0,  1.0,  2.0, 10.0]]),
    index=['A', 'B', 'C', 'D'],
    columns=['A', 'B', 'C', 'D'])


# In[23]:


glove_test_df = vsm.glove(glove_test_count_df, max_iter=1000, n=4)


# In[24]:


def correlation_test(true, pred):   
    mask = true > 0
    M = pred.dot(pred.T)
    with np.errstate(divide='ignore'):
        log_cooccur = np.log(true)
        log_cooccur[np.isinf(log_cooccur)] = 0.0
        row_log_prob = np.log(true.sum(axis=1))
        row_log_prob = np.outer(row_log_prob, np.ones(true.shape[1]))
        prob = log_cooccur - row_log_prob
    return np.corrcoef(prob[mask], M[mask])[0, 1]


# In[25]:


correlation_test(glove_test_count_df.values, glove_test_df.values)


# ### Applying GloVe to real VSMs

# The `vsm.glove` implementation is too slow to use on real matrices. The distribution in the `mittens` package is significantly faster, making its use possible even without a GPU (and it will be very fast indeed on a GPU machine):

# In[26]:


glove_model = GloVe()

imdb5_glv = glove_model.fit(imdb5.values)

imdb5_glv = pd.DataFrame(imdb5_glv, index=imdb5.index)


# In[27]:


vsm.neighbors('superb', imdb5_glv).head()


# ## Autoencoders
# 
# An autoencoder is a machine learning model that seeks to learn parameters that predict its own input. This is meaningful when there are intermediate representations that have lower dimensionality than the inputs. These provide a reduced-dimensional view of the data akin to those learned by LSA, but now we have a lot more design choices and a lot more potential to learn higher-order associations in the underyling data.

# ### Overview of the autoencoder method
# 
# The module `torch_autoencoder` uses PyToch to implement a simple one-layer autoencoder:
# 
# $$
# \begin{align}
# h &= \mathbf{f}(xW + b_{h}) \\
# \widehat{x} &= hW^{\top} + b_{x}
# \end{align}$$
# 
# Here, we assume that the hidden representation $h$ has a low dimensionality like 100, and that $\mathbf{f}$ is a non-linear activation function (the default for `TorchAutoencoder` is `tanh`). These are the major design choices internal to the network. It might also be meaningful to assume that there are two matrices of weights $W_{xh}$ and $W_{hx}$, rather than using $W^{\top}$ for the output step.
# 
# The objective function for autoencoders will implement some kind of assessment of the distance between the inputs and their predicted outputs. For example, one could use the one-half mean squared error:
# 
# $$\frac{1}{m}\sum_{i=1}^{m} \frac{1}{2}(\widehat{X[i]} - X[i])^{2}$$
# 
# where $X$ is the input matrix of examples (dimension $m \times n$) and $X[i]$ corresponds to the $i$th example.
# 
# When you call the `fit` method of `TorchAutoencoder`, it returns the matrix of hidden representations $h$, which is the new embedding space: same row count as the input, but with the column count set by the `hidden_dim` parameter.
# 
# For much more on autoencoders, see the 'Autoencoders' chapter of [Goodfellow et al. 2016](http://www.deeplearningbook.org).

# ### Testing the autoencoder implementation
# 
# Here's an evaluation that is meant to test the autoencoder implementation – we expect it to be able to full encode the input matrix because we know its rank is equal to the dimensionality of the hidden representation.

# In[28]:


def randmatrix(m, n, sigma=0.1, mu=0):
    return sigma * np.random.randn(m, n) + mu

def autoencoder_evaluation(nrow=1000, ncol=100, rank=20, max_iter=20000):
    """This an evaluation in which `TfAutoencoder` should be able
    to perfectly reconstruct the input data, because the
    hidden representations have the same dimensionality as
    the rank of the input matrix.
    """
    X = randmatrix(nrow, rank).dot(randmatrix(rank, ncol))
    ae = TorchAutoencoder(hidden_dim=rank, max_iter=max_iter)
    ae.fit(X)
    X_pred = ae.predict(X)
    mse = (0.5 * (X_pred - X)**2).mean()
    return(X, X_pred, mse)


# In[29]:


ae_max_iter = 100

_, _, ae = autoencoder_evaluation(max_iter=ae_max_iter)

print("Autoencoder evaluation MSE after {0} evaluations: {1:0.04f}".format(ae_max_iter, ae))


# ### Applying autoencoders to real VSMs
# 
# You can apply the autoencoder directly to the count matrix, but this could interact very badly with the internal activation function: if the counts are all very high or very low, then everything might get pushed irrevocably towards the extreme values of the activation.
# 
# Thus, it's a good idea to first normalize the values somehow. Here, I use `vsm.length_norm`:

# In[30]:


imdb5_l2 = imdb5.apply(vsm.length_norm, axis=1)


# In[31]:


imdb5_l2_ae = TorchAutoencoder(
    max_iter=100, hidden_dim=50, eta=0.001).fit(imdb5_l2)


# In[32]:


vsm.neighbors('superb', imdb5_l2_ae).head()


# This is very slow and seems not to work all that well. To speed things up, one can first apply LSA or similar:

# In[33]:


imdb5_l2_svd100 = vsm.lsa(imdb5_l2, k=100)


# In[34]:


imdb_l2_svd100_ae = TorchAutoencoder(
    max_iter=1000, hidden_dim=50, eta=0.01).fit(imdb5_l2_svd100)


# In[35]:


vsm.neighbors('superb', imdb_l2_svd100_ae).head()


# ## word2vec
# 
# The label __word2vec__ picks out a family of models in which the embedding for a word $w$ is trained to predict the words that co-occur with $w$. This intuition can be cashed out in numerous ways. Here, we review just the __skip-gram model__, due to [Mikolov et al. 2013](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality).

# ### Training data
# 
# The most natural starting point is to transform a corpus into a supervised data set by mapping each word to a subset (maybe all) of the words that it occurs with in a given window. Schematically:
# 
# __Corpus__: `it was the best of times, it was the worst of times, ...`
# 
# With window size 2:
# 
# ```
# (it, was)
# (it, the)
# (was, it)
# (was, the)
# (was, best)
# (the, was)
# (the, it)
# (the, best)
# (the, of)
# ...
# ```

# ### Basic skip-gram
# 
# The basic skip-gram model estimates the probability of an input–output pair $(a, b)$ as
# 
# $$P(b \mid a) = \frac{\exp(x_{a}w_{b})}{\sum_{b'\in V}\exp(x_{a}w_{b'})}$$
# 
# where $x_{a}$ is the row (word) vector representation of word $a$ and $w_{b}$ is the column (context) vector representation of word $b$. The objective is to minimize the following quantity:
# 
# $$
# -\sum_{i=1}^{m}\sum_{k=1}^{|V|}
# \textbf{1}\{c_{i}=k\} 
# \log
# \frac{
#     \exp(x_{i}w_{k})
# }{
#     \sum_{j=1}^{|V|}\exp(x_{i}w_{j})
# }$$
# 
# where $V$ is the vocabulary.
# 
# The inputs $x_{i}$ are the word representations, which get updated during training, and the outputs are one-hot vectors $c$. For example, if `was` is the 560th element in the vocab, then the output $c$ for the first example in the corpus above would be a vector of all $0$s except for a $1$ in the 560th position. $x$ would be the representation of `it` in the embedding space. 
# 
# The distribution over the entire output space for a given input word $a$ is thus a standard softmax classifier; here we add a bias term for good measure:
# 
# $$c = \textbf{softmax}(x_{a}W + b)$$
# 
# If we think of this model as taking the entire matrix $X$ as input all at once, then it becomes
# 
# $$c = \textbf{softmax}(XW + b)$$
# 
# and it is now very clear that we are back to the core insight that runs through all of our reweighting and dimensionality reduction methods: we have a word matrix $X$ and a context matrix $W$, and we are trying to push the dot products of these two embeddings in a specific direction: here, to maximize the likelihood of the observed co-occurrences in the corpus.

# ### Skip-gram with noise contrastive estimation 
# 
# Training the basic skip-gram model directly is extremely expensive for large vocabularies, because $W$, $b$, and the outputs $c$ get so large. 
# 
# A straightforward way to address this is to change the objective to use __noise contrastive estimation__ (negative sampling). Where $\mathcal{D}$ is the original training corpus and $\mathcal{D}'$ is a sample of pairs not in the corpus, we minimize
# 
# $$\sum_{a, b \in \mathcal{D}}-\log\sigma(x_{a}w_{b}) + \sum_{a, b \in \mathcal{D}'}\log\sigma(x_{a}w_{b})$$
# 
# with $\sigma$ the sigmoid activation function $\frac{1}{1 + \exp(-x)}$.
# 
# The advice of Mikolov et al. is to sample $\mathcal{D}'$ proportional to a scaling of the frequency distribution of the underlying vocabulary in the corpus:
# 
# $$P(w) = \frac{\textbf{count}(w)^{0.75}}{\sum_{w'\in V} \textbf{count}(w')}$$
# 
# where $V$ is the vocabulary.
# 
# Although this new objective function is a substantively different objective than the previous one, Mikolov et al. (2013) say that it should approximate it, and it is building on the same insight about words and their contexts. See [Levy and Golberg 2014](http://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization) for a proof that this objective reduces to PMI shifted by a constant value. See also [Cotterell et al. 2017](https://aclanthology.coli.uni-saarland.de/papers/E17-2028/e17-2028) for an interpretation of this model as a variant of PCA.

# ### word2vec resources
# 
# * In the usual presentation, word2vec training involves looping repeatedly over the sequence of tokens in the corpus, sampling from the context window from each word to create the positive training pairs. I assume that this same process could be modeled by sampling (row, column) index pairs from our count matrices proportional to their cell values. However, I couldn't get this to work well. I'd be grateful if someone got it work or figured out why it won't!
# 
# * Luckily, there are numerous excellent resources for word2vec. [The TensorFlow tutorial Vector representations of words](https://www.tensorflow.org/tutorials/word2vec) is very clear and links to code that is easy to work with. Because TensorFlow has a built in loss function called `tf.nn.nce_loss`, it is especially simple to define these models – one pretty much just sets up an embedding $X$, a context matrix $W$, and a bias $b$, and then feeds them plus a training batch to the loss function.
# 
# * The excellent [Gensim package](https://radimrehurek.com/gensim/) has an implementation that handles the scalability issues related to word2vec.

# ## Other methods
# 
# Learning word representations is one of the most active areas in NLP right now, so I can't hope to offer a comprehensive summary. I'll settle instead for identifying some overall trends and methods:
# 
# * The LexVec model of [Salle et al. 2016](https://aclanthology.coli.uni-saarland.de/papers/P16-2068/p16-2068) combines the core insight of GloVe (learn vectors that approximate PMI) with the insight from word2vec that we should additionally try to push words that don't appear together farther apart in the VSM. (GloVe simply ignores 0 count cells and so can't do this.)
# 
# * There is growing awareness that many apparently diverse models can be expressed as matrix factorization methods like SVD/LSA. See especially 
# [Singh and Gordon 2008](http://www.cs.cmu.edu/~ggordon/singh-gordon-unified-factorization-ecml.pdf),
# [Levy and Golberg 2014](http://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization), [Cotterell et al. 2017](https://www.aclweb.org/anthology/E17-2028/).
# 
# * Subword modeling ([reviewed briefly in the previous notebook](vsm_01_distributional.ipynb#Subword-information)) is increasingly yielding dividends. (It would already be central if most of NLP focused on languages with complex morphology!) Check out the papers at the Subword and Character-Level Models for NLP Workshops: [SCLeM 2017](https://sites.google.com/view/sclem2017/home), [SCLeM 2018](https://sites.google.com/view/sclem2018/home).
# 
# * Contextualized word representations have proven valuable in many contexts. These methods do not provide representations for individual words, but rather represent them in their linguistic context. This creates space for modeling how word senses vary depending on their context of use. We will study these methods later in the quarter, mainly in the context of identifying ways that might achieve better results on your projects.

# ## Exploratory exercises
# 
# These are largely meant to give you a feel for the material, but some of them could lead to projects and help you with future work for the course. These are not for credit.
# 
# 1. Try out some pipelines of reweighting, `vsm.lsa` at various dimensions, and `TorchAutoencoder` to see which seems best according to your sampling around with `vsm.neighbors` and high-level visualization with `vsm.tsne_viz`. Feel free to use other factorization methods defined in [sklearn.decomposition](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition) as well.
# 
# 1. What happens if you set `k=1` using `vsm.lsa`? What do the results look like then? What do you think this first (and now only) dimension is capturing?
# 
# 1. Modify `vsm.glove` so that it uses [the AdaGrad optimization method](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) as in the original paper. It's fine to use [the authors' implementation](http://nlp.stanford.edu/projects/glove/), [Jon Gauthier's implementation](http://www.foldl.me/2014/glove-python/), or the [mittens Numpy implementation](https://github.com/roamanalytics/mittens/blob/master/mittens/np_mittens.py) as references, but you might enjoy the challenge of doing this with no peeking at their code.
