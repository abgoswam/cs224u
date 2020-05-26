import nli
import os
import pandas as pd
import random
import json

mnli_dev_matched_tsv = r'C:\_misc\glue_data\MNLI\dev_matched.tsv'
anli2mnli_round1_dev_tsv = r'C:\_hackerreborn\cs224u\anli2mnli_round1\dev_matched.tsv'

df_mnli_dev_matched = pd.read_csv(mnli_dev_matched_tsv, sep='\t')
df_anli2mnli_round1_dev = pd.read_csv(anli2mnli_round1_dev_tsv, sep='\t')
