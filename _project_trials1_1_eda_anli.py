import nli
import os
import pandas as pd
import random
import json

DATA_HOME = os.path.join("data", "nlidata")

SNLI_HOME = os.path.join(DATA_HOME, "snli_1.0")

MULTINLI_HOME = os.path.join(DATA_HOME, "multinli_1.0")

ANNOTATIONS_HOME = os.path.join(DATA_HOME, "multinli_1.0_annotations")

ANLI_HOME = os.path.join(DATA_HOME, "anli_v0.1")

###################################################################
# ANLI Round 1 data only
def label_anli2mmnli(label):
    if label == "c":
        return "contradiction"
    elif label == "e":
        return "entailment"
    elif label == "n":
        return "neutral"
    else:
        raise IndexError

anli2mnli_round1_dev_jsonl_format = []

for ex in nli.ANLIDevReader(ANLI_HOME, rounds=(1,)).read():
    jsonl_format = {
        'annotator_labels': [],
        'genre': 'slate',
        'gold_label': label_anli2mmnli(ex.label),
        'pairID': ex.uid,
        'promptID': None,
        'sentence1': ex.context.rstrip(),
        'sentence1_binary_parse': None,
        'sentence1_parse': None,
        'sentence2': ex.hypothesis.rstrip(),
        'sentence2_binary_parse': None,
        'sentence2_parse': None
    }

    anli2mnli_round1_dev_jsonl_format.append(jsonl_format)

with open(r'C:\_hackerreborn\cs224u\anli2mnli_round1\dev_matched.jsonl', 'w', encoding='utf-8') as f:
    for item in anli2mnli_round1_dev_jsonl_format:
        s = json.dumps(item)
        f.write(s + "\n")

with open(r'C:\_hackerreborn\cs224u\anli2mnli_round1\dev_matched.tsv', 'w', encoding='utf-8') as f:
    f.write("index	promptID	pairID	genre	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2	label1	label2	label3	label4	label5	gold_label" + "\n")
    for i, item in enumerate(anli2mnli_round1_dev_jsonl_format):
        s = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\t{15}".format(
            i,
            item['promptID'],
            item['pairID'],
            item['genre'],
            item['sentence1_binary_parse'],
            item['sentence2_binary_parse'],
            item['sentence1_parse'],
            item['sentence2_parse'],
            item['sentence1'],
            item['sentence2'],
            None,
            None,
            None,
            None,
            None,
            item['gold_label'])

        f.write(s + "\n")

####################################################################
# MNLI data
multinli_labels_train = pd.Series(
    [ex.gold_label for ex in nli.MultiNLITrainReader(
        MULTINLI_HOME, filter_unlabeled=False).read()])

multinli_labels_train.value_counts()

multinli_labels_dev = pd.Series(
    [ex.gold_label for ex in nli.MultiNLIMatchedDevReader(
        MULTINLI_HOME, filter_unlabeled=True).read()])

multinli_labels_dev.value_counts()

nli.MultiNLIMatchedDevReader(MULTINLI_HOME, filter_unlabeled=True).read()

for line in open(os.path.join(MULTINLI_HOME, "multinli_1.0_dev_matched.jsonl"), encoding='utf8'):
    d = json.loads(line)
    break

# ANLI data
for r in (1,2,3):
    anli_labels_dev = pd.Series(
            [ex.label == ex.model_label for ex in nli.ANLIDevReader(ANLI_HOME, rounds=(r,)).read()]
        ).value_counts()

    print(anli_labels_dev)

    anli_labels_dev = pd.Series(
            [ex.label for ex in nli.ANLIDevReader(ANLI_HOME, rounds=(r,)).read()]
        ).value_counts()

    print(anli_labels_dev)

    print("-----------------")

#####################################################################



