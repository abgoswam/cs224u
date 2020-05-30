import torch
import nli
import os
import pandas as pd
import random
from fairseq.data.data_utils import collate_tokens

DATA_HOME = os.path.join("data", "nlidata")

SNLI_HOME = os.path.join(DATA_HOME, "snli_1.0")

MULTINLI_HOME = os.path.join(DATA_HOME, "multinli_1.0")

# Download RoBERTa already finetuned for MNLI
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')

roberta.cuda()

roberta.eval()  # disable dropout for evaluation


# p = [((ex.sentence1, ex.sentence2), ex.gold_label)
#      for ex in nli.MultiNLIMatchedDevReader(MULTINLI_HOME,
#                                             samp_percentage=1.0,
#                                             filter_unlabeled=False).read()]
#
# x, y = zip(*p)
# pd.Series(y).value_counts()


mnli_dev_matched_tsv = r'C:\_misc\glue_data\MNLI\dev_matched.tsv'
anli2mnli_round1_dev_tsv = r'C:\_misc\glue_data\ANLI_2_MNLI_R1\dev_matched.tsv'

# WARN : there is a problem loading up MNLI dev set this way in pandas
# There are some sentences as follows "abcd \t efgh"
# Pandas treats it correctly as 1 sentence
# But apparently MNLI dataset mandates it to be 2 sentences :  "abcd  efgh
df_mnli_dev_matched = pd.read_csv(mnli_dev_matched_tsv, sep='\t')
df_anli2mnli_round1_dev = pd.read_csv(anli2mnli_round1_dev_tsv, sep='\t')

mnli_dev_matched = df_mnli_dev_matched[['sentence1', 'sentence2']].values.tolist()

print("encoding..")

for ex in mnli_dev_matched:
    print(ex)
    break

X_dev = [roberta.encode(*ex) for ex in mnli_dev_matched]

mx = 0
mx_i = 0
for i, item in enumerate(X_dev):
    if len(item) > mx:
        mx = len(item)
        mx_i = i

print(len(X_dev), mx, mx_i)

# batch = collate_tokens(
#     [roberta.encode(pair[0], pair[1]) for pair in mnli_dev_matched], pad_idx=1
# )
#
# logprobs = roberta.predict('mnli', batch)