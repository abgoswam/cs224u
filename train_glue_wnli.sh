#!/bin/bash
# V2 fter inadvertently deleting V1.

echo "Abhishek Goswami - 1"

: '

# delete cached files
rm -rf /home/agoswami/data/glue_data/MNLI_WithWeakSupervision/cached_*

python run_glue.py \
--model_type bert \
--model_name_or_path bert-base-uncased \
--task_name MNLI \
--do_train \
--do_eval \
--do_lower_case \
--save_steps 5000 \
--data_dir /home/agoswami/data/glue_data/MNLI_WithWeakSupervision/ \
--max_seq_length 128 \
--per_gpu_train_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 3.0 \
--overwrite_output_dir \
--output_dir /home/agoswami/data/_output

rm -rf /home/agoswami/data/glue_data/MNLI_WithWeakSupervision/cached_*

python run_glue.py \
--model_type roberta \
--model_name_or_path roberta-base \
--task_name MNLI \
--do_train \
--do_eval \
--do_lower_case \
--save_steps 5000 \
--data_dir /home/agoswami/data/glue_data/MNLI_WithWeakSupervision/ \
--max_seq_length 128 \
--per_gpu_train_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 3.0 \
--overwrite_output_dir \
--output_dir /home/agoswami/data/_output_roberta

'

#: '

rm -rf /home/agoswami/data/glue_data/ANLI_A1/cached_*

python run_glue.py \
--model_type bert \
--model_name_or_path bert-base-uncased \
--task_name MNLI \
--do_eval \
--do_lower_case \
--data_dir /home/agoswami/data/glue_data/ANLI_A1/ \
--max_seq_length 128 \
--output_dir /home/agoswami/data/_output

rm -rf /home/agoswami/data/glue_data/ANLI_A1/cached_*

python run_glue.py \
--model_type roberta \
--model_name_or_path roberta-base \
--task_name MNLI \
--do_eval \
--do_lower_case \
--data_dir /home/agoswami/data/glue_data/ANLI_A1/ \
--max_seq_length 128 \
--output_dir /home/agoswami/data/_output_roberta

#'

echo "DONE"

