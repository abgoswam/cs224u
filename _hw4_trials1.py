from colors import ColorsCorpusReader
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch_color_describer import (
    ContextualColorDescriber, create_example_dataset)
import utils
from utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL


tiny_contexts, tiny_words, tiny_vocab = create_example_dataset(
    group_size=3, vec_dim=2)

toy_mod = ContextualColorDescriber(
    tiny_vocab,
    embedding=None,  # Option to supply a pretrained matrix as an `np.array`.
    embed_dim=10,
    hidden_dim=20,
    max_iter=100,
    eta=0.01,
    optimizer=torch.optim.Adam,
    batch_size=128,
    l2_strength=0.0,
    warm_start=False,
    device=None)

_ = toy_mod.fit(tiny_contexts, tiny_words)

metric = toy_mod.listener_accuracy(tiny_contexts, tiny_words)
print("listener_accuracy:", metric)

