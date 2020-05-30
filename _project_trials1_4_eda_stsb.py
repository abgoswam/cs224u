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

############### STSB ###############################################

stsb_train = r'C:\_misc\glue_data\STS-B\train.tsv'
df = pd.read_csv(stsb_train, sep='\t', quoting=csv.QUOTE_NONE)
n = df.shape[0]

# print(df.shape)
# print(n)
# print(df.head())

df_dict = df.to_dict()

def map_score_2_gold_label(score):
    if score < 2.0:
        return 'contradiction'
    elif score > 4.0:
        return 'entailment'
    else:
        return 'neutral'

with open(r'C:\_misc\glue_data\ANLI_A1_WithWeakSupervision\train_STSB.tsv', 'w', encoding='utf-8') as f:
    f.write(
        "index	promptID	pairID	genre	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2	label1	gold_label" + "\n")

    for i in range(n):
        s = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}".format(
            i,
            None,
            None,
            "STSB",
            None,
            None,
            None,
            None,
            df_dict['sentence1'][i],
            df_dict['sentence2'][i],
            None,
            map_score_2_gold_label(df_dict['score'][i]))

        f.write(s + "\n")

# verify
df_STSB = pd.read_csv(r'C:\_misc\glue_data\ANLI_A1_WithWeakSupervision\train_STSB.tsv', sep='\t', quoting=csv.QUOTE_NONE)
print(df_STSB.shape)
print(df_STSB['gold_label'].value_counts())

