import nli
import os
import pandas as pd
import random
import json
import csv

############### MNLI ###############################################

# verify
df_MNLI = pd.read_csv(r'C:\_misc\glue_data\ANLI_A1_WithWeakSupervision\train_MNLI.tsv', sep='\t', quoting=csv.QUOTE_NONE)
print(df_MNLI.shape)
print(df_MNLI['gold_label'].value_counts())

############### MRPC ###############################################

# verify
df_MRPC = pd.read_csv(r'C:\_misc\glue_data\ANLI_A1_WithWeakSupervision\train_MRPC.tsv', sep='\t', quoting=csv.QUOTE_NONE)
print(df_MRPC.shape)
print(df_MRPC['gold_label'].value_counts())

############### QQP ###############################################

# verify
df_QQP = pd.read_csv(r'C:\_misc\glue_data\ANLI_A1_WithWeakSupervision\train_QQP.tsv', sep='\t', quoting=csv.QUOTE_NONE)
print(df_QQP.shape)
print(df_QQP['gold_label'].value_counts())

############### STSB ###############################################

# verify
df_STSB = pd.read_csv(r'C:\_misc\glue_data\ANLI_A1_WithWeakSupervision\train_STSB.tsv', sep='\t', quoting=csv.QUOTE_NONE)
print(df_STSB.shape)
print(df_STSB['gold_label'].value_counts())

############### QNLI ###############################################

# verify
df_QNLI = pd.read_csv(r'C:\_misc\glue_data\ANLI_A1_WithWeakSupervision\train_QNLI.tsv', sep='\t', quoting=csv.QUOTE_NONE)
print(df_QNLI.shape)
print(df_QNLI['gold_label'].value_counts())

############### RTE ###############################################

# verify
df_RTE = pd.read_csv(r'C:\_misc\glue_data\ANLI_A1_WithWeakSupervision\train_RTE.tsv', sep='\t', quoting=csv.QUOTE_NONE)
print(df_RTE.shape)
print(df_RTE['gold_label'].value_counts())

############### WNLI ###############################################

# verify
df_WNLI = pd.read_csv(r'C:\_misc\glue_data\ANLI_A1_WithWeakSupervision\train_WNLI.tsv', sep='\t', quoting=csv.QUOTE_NONE)
print(df_WNLI.shape)
print(df_WNLI['gold_label'].value_counts())

########################################################################
df_train = pd.concat([df_MNLI, df_MRPC, df_QQP, df_STSB, df_QNLI, df_RTE, df_WNLI])
print(df_train.shape)

df_train_sample = df_train.sample(frac=0.1)
print(df_train_sample.shape)

output_file = r'C:\_misc\glue_data\ANLI_A1_WithWeakSupervision\train_MNLI_0.1_MRPC_0.1_QQP_0.1_STSB_0.1_QNLI_0.1_RTE_0.1_WNLI_0.1.tsv'
df_train_sample.to_csv(output_file, index = False, sep='\t')

df_train_sample2 = pd.read_csv(output_file, sep='\t', quoting=csv.QUOTE_NONE)
print(df_train_sample2.shape)
print(df_train_sample2['gold_label'].value_counts())