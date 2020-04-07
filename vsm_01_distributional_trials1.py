#!/usr/bin/env python
# coding: utf-8

# # Vector-space models: designs, distances, basic reweighting

# In[1]:


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


# ## Contents
# 
# 1. [Overview](#Overview)
# 1. [Motivation](#Motivation)
# 1. [Terminological notes](#Terminological-notes)
# 1. [Set-up](#Set-up)
# 1. [Matrix designs](#Matrix-designs)
# 1. [Pre-computed example matrices](#Pre-computed-example-matrices)
# 1. [Vector comparison](#Vector-comparison)
#   1. [Euclidean](#Euclidean)
#   1. [Length normalization](#Length-normalization)
#   1. [Cosine distance](#Cosine-distance)
#   1. [Matching-based methods](#Matching-based-methods)
#   1. [Summary](#Summary)
# 1. [Distributional neighbors](#Distributional-neighbors)
# 1. [Matrix reweighting](#Matrix-reweighting)
#   1. [Normalization](#Normalization)
#   1. [Observed/Expected](#Observed/Expected)
#   1. [Pointwise Mutual Information](#Pointwise-Mutual-Information)
#   1. [TF-IDF](#TF-IDF)
# 1. [Subword information](#Subword-information)
# 1. [Visualization](#Visualization)
# 1. [Exploratory exercises](#Exploratory-exercises)

# ## Overview
# 
# This notebook is the first in our series about creating effective __distributed representations__. The focus is on matrix designs, assessing similarity, and methods for matrix reweighting.
# 
# The central idea (which takes some getting used to!) is that we can represent words and phrases as dense vectors of real numbers. These take on meaning by being __embedded__ in a larger matrix of representations with comparable structure.

# ## Motivation
# 
# Why build distributed representations? There are potentially many reasons. The two we will emphasize in this course:
# 
# 1. __Understanding words in context__: There is value to linguists in seeing what these data-rich approaches can teach us about natural language lexicons, and there is value for social scientists in understanding how words are being used.
# 
# 1. __Feature representations for other models__: As we will see, many models can benefit from representing examples as distributed representations.

# ## Terminological notes

# * The distributed representations we build will always be vectors of real numbers. The models are often called __vector space models__ (VSMs).
# 
# * __Distributional representations__ are the special case where the data come entirely from co-occurrence counts in corpora. 
# 
# * We'll look at models that use supervised labels to obtain vector-based word representations. These aren't purely distributional, in that they take advantage of more than just co-occurrence patterns among items in the vocabulary, but they share the idea that words can be modeled with vectors.
# 
# * If a neural network is used to train the representations, then they might be called __neural representations__.
# 
# * The term __word embedding__ is also used for distributed representations, including distributional ones. This term is a reminder that vector representations are meaningful only when embedded in and compared with others in a unified space (usually a matrix) of representations of the same type.
# 
# * In any case, __distributed representation__ seems like the most general cover term for what we're trying to achieve, and its only downside is that sometimes people think it has something to do with distributed databases.

# ## Set-up
# 
# * Make sure your environment meets all the requirements for [the cs224u repository](https://github.com/cgpotts/cs224u/). For help getting set-up, see [setup.ipynb](setup.ipynb).
# 
# * Download [the course data](http://web.stanford.edu/class/cs224u/data/data.tgz), unpack it, and place it in the directory containing the course repository – the same directory as this notebook. (If you want to put it somewhere else, change `DATA_HOME` below.)

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import os
import pandas as pd
import vsm


# In[3]:


DATA_HOME = os.path.join('data', 'vsmdata')


# ## Matrix designs
# 
# There are many, many ways to define distributional matrices. Here's a schematic overview that highlights the major decisions for building a word $\times$ word matrix:
# 
# 1. Define a notion of __co-occurrence context__. This could be an entire document, a paragraph, a sentence, a clause, an NP — whatever domain seems likely to capture the associations you care about.
# 
# 1. Define a __count scaling method__. The simplest method just counts everything in the context window, giving equal weight to everything inside it. A common alternative is to scale the weights based on proximity to the target word – e.g., $1/d$, where $d$ is the distance in tokens from the target.
# 
# 1. Scan through your corpus building a dictionary $d$ mapping word-pairs to co-occurrence values. Every time a pair of words $w$ and $w'$ occurs in the same context (as you defined it in 1), increment $d[(w, w')]$ by whatever value is determined by your weighting scheme. You'd increment by $1$ with the weighting scheme that simply counts co-occurrences.
# 
# 1. Using the count dictionary $d$ that you collected in 3, establish your full vocabulary $V$, an ordered list of words types. 
#     1. For large collections of documents, $|V|$ will typically be huge. You will probably want to winnow the vocabulary at this point. 
#     1. You might do this by filtering to a specific subset, or just imposing a minimum count threshold. 
#     1. You might impose a minimum count threshold even if $|V|$ is small — for words with very low counts, you simply don't have enough evidence to support good representations.
#     1. For words outside the vocabulary you choose, you could ignore them entirely or accumulate all their values into a designated _UNK_ vector.
# 
# 1. Now build a matrix $M$ of dimension $|V| \times |V|$. Both the rows and the columns of $M$ represent words. Each cell $M[i, j]$ is filled with the value $d[(w_{1}, w_{j})]$.

# ## Pre-computed example matrices
# 
# The data distribution includes four matrices that we'll use for hands-on exploration. All of them were designed in the same basic way:
# 
# * They are word $\times$ word matrices with 5K rows and 5K columns. 
# 
# * The vocabulary is the top 5K most frequent unigrams.
# 
# Two come from IMDB user-supplied movie reviews, and two come from Gigaword, a collection of newswire and newspaper text. Further details:
# 
# |filename | source | window size| count weighting |
# |---------|--------|------------|-----------------|
# |imdb_window5-scaled.csv.gz | IMDB movie reviews | 5| 1/d |
# |imdb_window20-flat.csv.gz | IMDB movie reviews | 20| 1 |
# |gigaword_window5-scaled.csv.gz | Gigaword | 5 | 1/d |
# |gigaword_window20-flat.csv.gz | Gigaword | 20 | 1 |
# 
# Any hunches about how these matrices might differ from each other?

# In[4]:


imdb5 = pd.read_csv(
    os.path.join(DATA_HOME, 'imdb_window5-scaled.csv.gz'), index_col=0)


# In[5]:


imdb20 = pd.read_csv(
    os.path.join(DATA_HOME, 'imdb_window20-flat.csv.gz'), index_col=0)


# In[6]:


giga5 = pd.read_csv(
    os.path.join(DATA_HOME, 'giga_window5-scaled.csv.gz'), index_col=0)


# In[7]:


giga20 = pd.read_csv(
    os.path.join(DATA_HOME, 'giga_window20-flat.csv.gz'), index_col=0)


# ## Vector comparison
# 
# Vector comparisons form the heart of our analyses in this context. 
# 
# * For the most part, we are interested in measuring the __distance__ between vectors. The guiding idea is that semantically related words should be close together in the vector spaces we build, and semantically unrelated words should be far apart.
# 
# * The [scipy.spatial.distance](http://docs.scipy.org/doc/scipy-0.14.0/reference/spatial.distance.html) module has a lot of vector comparison methods, so you might check them out if you want to go beyond the functions defined and explored here. Read the documentation closely, though: many of those methods are defined only for binary vectors, whereas the VSMs we'll use allow all float values.

# ### Euclidean
# 
# The most basic and intuitive distance measure between vectors is __euclidean distance__. The euclidean distance between two vectors $u$ and $v$ of dimension $n$ is
# 
# $$\textbf{euclidean}(u, v) = 
# \sqrt{\sum_{i=1}^{n}|u_{i} - v_{i}|^{2}}$$
# 
# In two-dimensions, this corresponds to the length of the most direct line between the two points.
# 
# In `vsm.py`, the function `euclidean` just uses the corresponding [scipy.spatial.distance](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html) method to define it.

# Here's the tiny vector space from the screencast on vector comparisons associated with this notebook:

# In[8]:


ABC = pd.DataFrame([
    [ 2.0,  4.0], 
    [10.0, 15.0], 
    [14.0, 10.0]],
    index=['A', 'B', 'C'],
    columns=['x', 'y'])    


# In[9]:


ABC


# In[10]:


def plot_ABC(df):
    ax = df.plot.scatter(x='x', y='y', marker='.', legend=False)
    m = df.values.max(axis=None)
    ax.set_xlim([0, m*1.2])
    ax.set_ylim([0, m*1.2])
    for label, row in df.iterrows():
        ax.text(row['x'], row['y'], label)


# In[11]:


plot_ABC(ABC)


# The euclidean distances align well with raw visual distance in the plot:

# In[12]:


def abc_comparisons(df, distfunc):
    for a, b in (('A', 'B'), ('B', 'C')):
        dist = distfunc(df.loc[a], df.loc[b])
        print('{0:}({1:}, {2:}) = {3:7.02f}'.format(
            distfunc.__name__, a, b, dist))


# In[13]:


abc_comparisons(ABC, vsm.euclidean)


# However, suppose we think of the vectors as word meanings in the vector-space sense. In that case, the values don't look good: 
# 
# * The distributions of B and C are more or less directly opposed, suggesting very different meanings, whereas A and B are rather closely aligned, abstracting away from the fact that the first is far less frequent than the second. 
# 
# * In terms of the large models we will soon explore, A and B resemble a pair like _superb_ and _good_, which have similar meanings but very different frequencies. 
# 
# * In contrast, B and C are like _good_ and _disappointing_ — similar overall frequencies but different distributions with respect to the overall vocabulary.

# ### Length normalization
# 
# These affinities are immediately apparent if we __normalize the vectors by their length__. To do this, we first define the L2-length of a vector:
# 
# $$\|u\|_{2} = \sqrt{\sum_{i=1}^{n} u_{i}^{2}}$$
# 
# And then the normalization step just divides each value by this quantity:
# 
# $$\left[ 
#   \frac{u_{1}}{\|u\|_{2}}, 
#   \frac{u_{2}}{\|u\|_{2}}, 
#   \ldots 
#   \frac{u_{n}}{\|u\|_{2}} 
#  \right]$$

# In[14]:


ABC_normed = ABC.apply(vsm.length_norm, axis=1)


# In[15]:


plot_ABC(ABC_normed)    


# In[16]:


abc_comparisons(ABC_normed, vsm.euclidean)


# Here, the connection between A and B is more apparent, as is the opposition between B and C.

# ### Cosine distance
# 
# Cosine distance takes overall length into account. The cosine distance between two vectors $u$ and $v$ of dimension $n$ is
# 
# $$\textbf{cosine}(u, v) = 
# 1 - \frac{\sum_{i=1}^{n} u_{i} \cdot v_{i}}{\|u\|_{2} \cdot \|v\|_{2}}$$
# 
# The similarity part of this (the righthand term of the subtraction) is actually measuring the angles between the two vectors. The result is the same (in terms of rank order) as one gets from first normalizing both vectors using $\|\cdot\|_{2}$ and then calculating their Euclidean distance.

# In[17]:


abc_comparisons(ABC, vsm.cosine)


# So, in building in the length normalization, cosine distance achieves our goal of associating A and B and separating both from C.

# ### Matching-based methods
# 
# Matching-based methods are also common in the literature. The basic matching measure effectively creates a vector consisting of all of the smaller of the two values at each coordinate, and then sums them:
# 
# $$\textbf{matching}(u, v) = \sum_{i=1}^{n} \min(u_{i}, v_{i})$$
# 
# This is implemented in `vsm` as `matching`.
# 
# One approach to normalizing the matching values is the [__Jaccard coefficient__](https://en.wikipedia.org/wiki/Jaccard_index). The numerator is the matching coefficient. The denominator — the normalizer — is intuitively like the set union: for binary vectors, it gives the cardinality of the union of the two being compared:
# 
# $$\textbf{jaccard}(u, v) = 
# 1 - \frac{\textbf{matching}(u, v)}{\sum_{i=1}^{n} \max(u_{i}, v_{i})}$$

# ### Summary
# 
# Suppose we set for ourselves the goal of associating A with B and disassociating B from C, in keeping with the semantic intuition expressed above. Then we can assess distance measures by whether they achieve this goal:

# In[18]:


for m in (vsm.euclidean, vsm.cosine, vsm.jaccard):
    fmt = {
        'n': m.__name__,  
        'AB': m(ABC.loc['A'], ABC.loc['B']), 
        'BC': m(ABC.loc['B'], ABC.loc['C'])}
    print('{n:>15}(A, B) = {AB:5.2f} {n:>15}(B, C) = {BC:5.2f}'.format(**fmt))


# ## Distributional neighbors
# 
# The `neighbors` function in `vsm` is an investigative aide. For a given word `w`, it ranks all the words in the vocabulary according to their distance from `w`, as measured by `distfunc` (default: `vsm.cosine`).
# 
# By playing around with this function, you can start to get a sense for how the distance functions differ. Here are some example uses; you might try some new words to get a feel for what these matrices are like and how different words look.

# In[19]:


vsm.neighbors('A', ABC, distfunc=vsm.euclidean)


# In[20]:


vsm.neighbors('A', ABC, distfunc=vsm.cosine)


# In[21]:


vsm.neighbors('good', imdb5, distfunc=vsm.euclidean).head()


# In[22]:


vsm.neighbors('good', imdb20, distfunc=vsm.euclidean).head()


# In[23]:


vsm.neighbors('good', imdb5, distfunc=vsm.cosine).head()


# In[24]:


vsm.neighbors('good', imdb20, distfunc=vsm.cosine).head()


# In[25]:


vsm.neighbors('good', giga20, distfunc=vsm.cosine).head()


# ## Matrix reweighting
# 
# * The goal of reweighting is to amplify the important, trustworthy, and unusual, while deemphasizing the mundane and the quirky. 
# 
# * Absent a defined objective function, this will remain fuzzy, but the intuition behind moving away from raw counts is that frequency is a poor proxy for our target semantic ideas.

# ### Normalization
# 
# Normalization (row-wise or column-wise) is perhaps the simplest form of reweighting. With `vsm.length_norm`, we normalize using `vsm.vector_length`. We can also normalize each row by the sum of its values, which turns each row into a probability distribution over the columns:
# 
# $$\left[ 
#   \frac{u_{1}}{\sum_{i=1}^{n}u_{i}}, 
#   \frac{u_{2}}{\sum_{i=1}^{n}u_{i}}, 
#   \ldots
#   \frac{u_{n}}{\sum_{i=1}^{n}u_{i}}, 
# \right]$$
# 
# These normalization measures are __insensitive to the magnitude of the underlying counts__. This is often a mistake in the messy world of large data sets; $[1,10]$ and $[1000,10000]$ are very different vectors in ways that will be partly or totally obscured by normalization.

# ### Observed/Expected
# 
# Reweighting by observed-over-expected values captures one of the central patterns in all of VSMs: we can adjust the actual cell value in a co-occurrence matrix using information from the corresponding row and column. 
# 
# In the case of observed-over-expected, the rows and columns define our expectation about what the cell value would be if the two co-occurring words were independent. In dividing the observed count by this value, we amplify cells whose values are larger than we would expect.
# 
# So that this doesn't look more complex than it is, for an $m \times n$ matrix $X$, define
# 
# $$\textbf{rowsum}(X, i) = \sum_{j=1}^{n}X_{ij}$$
# 
# $$\textbf{colsum}(X, j) = \sum_{i=1}^{m}X_{ij}$$
# 
# $$\textbf{sum}(X) = \sum_{i=1}^{m}\sum_{j=1}^{n} X_{ij}$$
# 
# $$\textbf{expected}(X, i, j) = 
# \frac{
#   \textbf{rowsum}(X, i) \cdot \textbf{colsum}(X, j)
# }{
#   \textbf{sum}(X)
# }$$
# 
# 
# Then the observed-over-expected value is
# 
# $$\textbf{oe}(X, i, j) = \frac{X_{ij}}{\textbf{expected}(X, i, j)}$$
# 
# In many contexts, it is more intuitive to first normalize the count matrix into a joint probability table and then think of $\textbf{rowsum}$ and $\textbf{colsum}$ as probabilities. Then it is clear that we are comparing the observed joint probability with what we would expect it to be under a null hypothesis of independence. These normalizations do not affect the final results, though.
# 
# Let's do a quick worked-out example. Suppose we have the count matrix $X$ = 
# 
# |    .     | a  | b  | rowsum |
# |----------|----|----|-------|
# | __x__    | 34 | 11 |  45   |
# | __y__    | 47 | 7  |  54   |
# |__colsum__| 81 | 18 |  99   |
# 
# Then we calculate like this:
# 
# $$\textbf{oe}(X, 1, 0) = \frac{47}{\frac{54 \cdot 81}{99}} = 1.06$$
# 
# And the full table looks like this:
# 
# |    .   | a    | b    | 
# |--------|------|------|
# | __x__  | 0.92 | 1.34 | 
# | __y__  | 1.06 | 0.71 |

# In[26]:


oe_ex = np.array([[ 34.,  11.], [ 47.,   7.]])

vsm.observed_over_expected(oe_ex).round(2)


# The implementation `vsm.observed_over_expected` should be pretty efficient.

# In[27]:


imdb5_oe = vsm.observed_over_expected(imdb5)


# In[28]:


imdb20_oe = vsm.observed_over_expected(imdb20)


# In[29]:


vsm.neighbors('good', imdb5_oe).head()


# In[30]:


vsm.neighbors('good', imdb20_oe).head()


# ### Pointwise Mutual Information
# 
# Pointwise Mutual Information (PMI) is observed-over-expected in log-space:
# 
# $$\textbf{pmi}(X, i, j) = \log\left(\frac{X_{ij}}{\textbf{expected}(X, i, j)}\right)$$
# 
# This basic definition runs into a problem for $0$ count cells. The usual response is to set $\log(0) = 0$, but this is arguably confusing – cell counts that are smaller than expected get negative values, cell counts that are larger than expected get positive values, and 0-count values are placed in the middle of this ranking without real justification.
# 
# For this reason, it is more typical to use __Positive PMI__, which maps all negative PMI values to $0$:
# 
# $$\textbf{ppmi}(X, i, j) = 
# \begin{cases}
# \textbf{pmi}(X, i, j) & \textrm{if } \textbf{pmi}(X, i, j) > 0 \\
# 0 & \textrm{otherwise}
# \end{cases}$$
# 
# This is the default for `vsm.pmi`.

# In[31]:


imdb5_pmi = vsm.pmi(imdb5)


# In[32]:


imdb20_pmi = vsm.pmi(imdb20)


# In[33]:


vsm.neighbors('good', imdb5_pmi).head()


# In[34]:


vsm.neighbors('good', imdb20_pmi).head()


# In[35]:


giga20_pmi = vsm.pmi(giga20)


# In[36]:


vsm.neighbors('market', giga20_pmi).head()


# ### TF-IDF
# 
# Perhaps the best known reweighting schemes is __Term Frequency–Inverse Document Frequency (TF-IDF)__, which is, I believe, still the backbone of today's Web search technologies. As the name suggests, it is built from TF and IDF measures:
# 
# For an $m \times n$ matrix $X$:
# 
# $$\textbf{TF}(X, i, j) = \frac{X_{ij}}{\textbf{colsum}(X, i, j)}$$
# 
# $$\textbf{IDF}(X, i, j) = \log\left(\frac{n}{|\{k : X_{ik} > 0\}|}\right)$$
# 
# $$\textbf{TF-IDF}(X, i, j) = \textbf{TF}(X, i, j) \cdot \textbf{IDF}(X, i, j)$$
# 
# 
# TF-IDF generally performs best with sparse matrices. It severely punishes words that appear in many documents; if a word appears in every document, then its IDF value is 0. As a result, it can even be problematic with verb dense word $\times$ word matrices like ours, where most words appear with most other words.
# 
# There is an implementation of TF-IDF for dense matrices in `vsm.tfidf`.
# 
# __Important__: `sklearn`'s version, [TfidfTransformer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer), assumes that term frequency (TF) is defined row-wise and document frequency is defined column-wise. That is, it assumes `sklearn`'s document $\times$ word basic design, which makes sense for classification tasks, where the design is example $\times$ features. This is the transpose of the way we've been thinking.

# ## Subword information
# 
# [Schütze (1993)](https://papers.nips.cc/paper/603-word-space) pioneered the use of subword information to improve representations by reducing sparsity, thereby increasing the density of connections in a VSM. In recent years, this idea has shown value in numerous contexts. 
# 
# [Bojanowski et al. (2016)](https://arxiv.org/abs/1607.04606) (the [fastText](https://fasttext.cc) team) explore a particularly straightforward approach to doing this: represent each word as the sum of the representations for the character-level n-grams it contains.
# 
# It is simple to derive character-level n-gram representations from our existing VSMs. The function `vsm.ngram_vsm` implements the basic step. Here, we create the 4-gram version of `imdb5`:

# In[37]:


imdb5_ngrams = vsm.ngram_vsm(imdb5, n=4)


# In[38]:


imdb5_ngrams.shape


# This has the same column dimension as the `imdb5`, but the rows are expanded with all the 4-grams, including boundary symbols `<w>` and `</w>`. 
# 
# `vsm.character_level_rep` is a simple function for creating new word representations from the associated character-level ones. Many variations on that function are worth trying – for example, you could include the original word vector where available, change the aggregation method from `sum` to something else, use a real morphological parser instead of just n-grams, and so on.

# One very powerful thing about this is that we can represent words that are not even in the original VSM:

# In[39]:


'superbly' in imdb5.index


# In[40]:


superbly = vsm.character_level_rep("superbly", imdb5_ngrams)


# In[41]:


superb = vsm.character_level_rep("superb", imdb5_ngrams)


# In[42]:


vsm.cosine(superb, superbly)


# ## Visualization
# 
# * You can begin to get a feel for what your matrix is like by poking around with `vsm.neighbors` to see who is close to or far from whom. 
# 
# * It's very useful to complement this with the more holistic view one can get from looking at a visualization of the entire vector space. 
# 
# * Of course, any visualization will have to be much, much lower dimension than our actual VSM, so we need to proceed cautiously, balancing the high-level view with more fine-grained exploration.
# 
# * We won't have time this term to cover VSM visualization in detail. scikit-learn has a bunch of functions for doing this in [sklearn.manifold](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold), and the [user guide](http://scikit-learn.org/stable/modules/manifold.html#manifold-learning) for that package is detailed.
# 
# * It's also worth checking out the online TensorFlow [Embedding Projector tool](http://projector.tensorflow.org), which includes a fast implementation of t-SNE.
# 
# * In addition, `vsm.tsne_viz` is a wrapper around [sklearn.manifold.TSNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE) that handles the basic preprocessing and layout for you. t-SNE stands for [t-Distributed Stochastic Neighbor Embedding](http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf), a powerful method for visualizing high-dimensional vector spaces in 2d. See also [Multiple Maps t-SNE](https://lvdmaaten.github.io/multiplemaps/Multiple_maps_t-SNE/Multiple_maps_t-SNE.html).

# In[43]:


vsm.tsne_viz(imdb20_pmi, random_state=42)


# ## Exploratory exercises
# 
# These are largely meant to give you a feel for the material, but some of them could lead to projects and help you with future work for the course. These are not for credit.
# 
# 1. Recall that there are two versions each of the IMDB and Gigaword matrices: one with window size 5 and counts scaled as $1/d$ where $d$ is the distance from the target word; and one with a window size of 20 and no scaling of the values. Using `vsm.neighbors` to explore, how would you describe the impact of these different designs?
# 
# 1. IMDB and Gigaword are very different domains. Using `vsm.neighbors`, can you find cases where the dominant sense of a word is clearly different in the two domains in a way that is reflected by vector-space proximity?
# 
# 1. We saw that euclidean distance favors raw frequencies. Find words in the matrix `imdb20` that help make this point: a pair that are semantically unrelated but close according to `vsm.euclidean`, and a pair that are semantically related by far apart according to `vsm.euclidean`.
# 
# 1. Run 
# 
#   ```amod = pd.read_csv(os.path.join(DATA_HOME, 'gigawordnyt-advmod-matrix.csv.gz'), index_col=0)``` 
#   
#   to read in an adjective $\times$ adverb matrix derived from the Gigaword corpus. Each cell contains the number of times that the modifier phrase __ADV ADJ__ appeared in Gigaword as given by dependency parses of the data. __ADJ__ is the row value and __ADV__ is the column value. Using the above techniques and measures, try to get a feel for what can be done with this matrix.
# 
# 1. [Turney and Pantel (2010)](https://jair.org/index.php/jair/article/view/10640), p. 158, propose a "contextual discounting" extension of PMI to try to address its bias for low-frequency events. Extend `vsm.pmi` so that the user has the option of performing this discounting with the keyword argument `discounting=True`.
