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

############### WNLI ###############################################

wnli_train = r'C:\_misc\glue_data\WNLI\train.tsv'
df = pd.read_csv(wnli_train, sep='\t', quoting=csv.QUOTE_NONE)
n = df.shape[0]

# print(df.shape)
# print(n)
# print(df.head())

df_dict = df.to_dict()

map_label_2_gold_label = {1: 'entailment', 0: 'contradiction'}

with open(r'C:\_misc\glue_data\ANLI_A1_WithWeakSupervision\train_WNLI.tsv', 'w', encoding='utf-8') as f:
    f.write(
        "index	promptID	pairID	genre	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2	label1	gold_label" + "\n")

    for i in range(n):
        s = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}".format(
            i,
            None,
            None,
            "WNLI",
            None,
            None,
            None,
            None,
            df_dict['sentence1'][i],
            df_dict['sentence2'][i],
            None,
            map_label_2_gold_label[df_dict['label'][i]])

        f.write(s + "\n")

# verify
df_WNLI = pd.read_csv(r'C:\_misc\glue_data\ANLI_A1_WithWeakSupervision\train_WNLI.tsv', sep='\t', quoting=csv.QUOTE_NONE)
print(df_WNLI.shape)
print(df_WNLI['gold_label'].value_counts())

