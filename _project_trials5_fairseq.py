import torch
import nli
import os
from sklearn.metrics import classification_report
import collections

DATA_HOME = os.path.join("data", "nlidata")

SNLI_HOME = os.path.join(DATA_HOME, "snli_1.0")

MULTINLI_HOME = os.path.join(DATA_HOME, "multinli_1.0")

ANLI_HOME = os.path.join(DATA_HOME, "anli_v0.1")

##################################################################
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.cuda()
roberta.eval()

dev = [((ex.sentence1, ex.sentence2), ex.gold_label)
       for ex in nli.MultiNLIMatchedDevReader(MULTINLI_HOME, samp_percentage=.1).read()]

# dev = [((ex.sentence1, ex.sentence2), ex.gold_label)
#        for ex in nli.MultiNLIMismatchedDevReader(MULTINLI_HOME, samp_percentage=.1).read()]

dev = [((ex.context, ex.hypothesis), ex.label)
       for ex in nli.ANLIDevReader(ANLI_HOME, rounds=(1,)).read()]

X_dev_str, y_dev = zip(*dev)
print(collections.Counter(y_dev))

print("encoding..")
X_dev = [roberta.encode(*ex) for ex in X_dev_str]

print("predicting..")
pred_indices = [roberta.predict('mnli', ex).argmax() for ex in X_dev]

to_str = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
# to_str = {0: 'c', 1: 'n', 2: 'e'}

preds = [to_str[c.item()] for c in pred_indices]

print(classification_report(y_dev, preds))

###################################################################
# batch = collate_tokens(
#     [roberta.encode(*ex) for ex in X_dev_str], pad_idx=1
# )
#
# logprobs = roberta.predict('mnli', batch)

###################################################################
# import torch
# from fairseq.data.data_utils import collate_tokens
#
# roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
# roberta.eval()
#
# batch_of_pairs = [
#     ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
#     ['Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.'],
#     ['potatoes are awesome.', 'I like to run.'],
#     ['Mars is very far from earth.', 'Mars is very close. Mars is very close. Mars is very close.'*10000],
# ]
#
# batch = collate_tokens(
#     [roberta.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1
# )
#
# logprobs = roberta.predict('mnli', batch)
# print(logprobs.argmax(dim=1))
#
# tokens = roberta.encode('Mars is very far from earth.', 'Mars is very close. Mars is very close. Mars is very close.'*10000)
