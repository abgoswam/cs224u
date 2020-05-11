from colors import ColorsCorpusReader
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch_color_describer import (
    ContextualColorDescriber, create_example_dataset,
    Encoder,
    Decoder,
    EncoderDecoder)
import utils
import torch.nn as nn
from utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL


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

        # V1
        # emb_with_colors = torch.empty(m, k, self.embed_dim + self.color_dim)  # shape : (m, k, embed_dim + color_dim)
        # for i in range(k):
        #     # emb[:, i, :] shape (m, embed_dim)
        #     # target_colors shape (m, color_dim)
        #     emb_with_colors[:, i, :] = torch.cat((emb[:, i, :], target_colors), 1)
        #
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # return emb_with_colors.to(device)

        # V2
        p = target_colors.unsqueeze(1)  # add additional dimension.  p : (m, 1, color_dim)
        p = p.repeat_interleave(k, 1)  # repeat the *elements* of tensor along dimension 1.  p : (m, k, color_dim)
        emb2 = torch.cat((emb, p), 2)  # torch.cat  emb2 : (m, k, embed_dim + color_dim)
        return emb2

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


tiny_contexts, tiny_words, tiny_vocab = create_example_dataset(
    group_size=3, vec_dim=2)

toy_mod = ColorizedInputDescriber(
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

