import os
import sst
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from torch_rnn_classifier import TorchRNNClassifier, TorchRNNClassifierModel
from torch_rnn_classifier import TorchRNNClassifier
from sklearn.metrics import classification_report
import utils

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Set all the random seeds for reproducibility. Only the
# system and torch seeds are relevant for this notebook.

utils.fix_random_seeds()

SST_HOME = os.path.join("data", "trees")

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

hf_fine_tune_mod = HfBertClassifier(
    'bert-base-cased',
    batch_size=16, # Crucial; large batches will eat up all your memory!
    max_iter=4,
    eta=0.00002)

hf_train = list(sst.train_reader(SST_HOME, class_func=sst.ternary_class_func))
hf_dev = list(sst.dev_reader(SST_HOME, class_func=sst.ternary_class_func))

hf_train_tiny = hf_train[:32]
hf_dev_tiny = hf_dev[:32]

X_hf_tree_train, y_hf_train = zip(*hf_train_tiny)
X_hf_tree_dev, y_hf_dev = zip(*hf_dev_tiny)

X_hf_str_train = [" ".join(tree.leaves()) for tree in X_hf_tree_train]
X_hf_str_dev = [" ".join(tree.leaves()) for tree in X_hf_tree_dev]

X_hf_indices_train = hf_fine_tune_mod.encode(X_hf_str_train)
X_hf_indices_dev = hf_fine_tune_mod.encode(X_hf_str_dev)

_ = hf_fine_tune_mod.fit(X_hf_indices_train, y_hf_train)