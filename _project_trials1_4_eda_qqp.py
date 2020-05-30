import nli
import os
import pandas as pd
import random
import json
import csv
import numpy as np

############### MNLI ###############################################

# verify
df_MNLI = pd.read_csv(r'C:\_misc\glue_data\ANLI_A1_WithWeakSupervision\train_MNLI.tsv', sep='\t', quoting=csv.QUOTE_NONE)
print(df_MNLI.shape)

############### QQP ###############################################

qqp_train = r'C:\_misc\glue_data\QQP\train.tsv'
df = pd.read_csv(qqp_train, sep='\t', quoting=csv.QUOTE_NONE)
n = df.shape[0]

# print(df.shape)
# print(n)
# print(df.head())

df_dict = df.to_dict()
map_is_duplicate_2_gold_label = {0: 'contradiction', 1: 'entailment'}

with open(r'C:\_misc\glue_data\ANLI_A1_WithWeakSupervision\train_QQP.tsv', 'w', encoding='utf-8') as f:
    f.write(
        "index	promptID	pairID	genre	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2	label1	gold_label" + "\n")

    for i in range(n):
        # see line 36733 in file train.tsv for QQP dataset.
        if np.isnan(df_dict['is_duplicate'][i]):
            print("Skipping : ", i)
            continue

        is_dup = int(df_dict['is_duplicate'][i])

        s = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}".format(
            i,
            None,
            None,
            "QQP",
            None,
            None,
            None,
            None,
            df_dict['question1'][i],
            df_dict['question2'][i],
            None,
            map_is_duplicate_2_gold_label[is_dup])

        f.write(s + "\n")

# verify
df_QQP = pd.read_csv(r'C:\_misc\glue_data\ANLI_A1_WithWeakSupervision\train_QQP.tsv', sep='\t', quoting=csv.QUOTE_NONE)
print(df_QQP.shape)

