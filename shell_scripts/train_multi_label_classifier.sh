#!/usr/bin/env bash

PROJECT_PATH=$(cd $(dirname $(cd $(dirname $0); pwd)); pwd)


export CUDA_VISIBLE_DEVICES=1
export BERT_BASE_DIR=${PROJECT_PATH}/model_data/bert/ernie

echo "Use cuda device ${CUDA_VISIBLE_DEVICES}"
echo "Bert pretrained model is in ${BERT_BASE_DIR}"

tag=biz1

python -m model_train.bert.multi_label_bert_classifier \
  --task_name=my_business \
  --biz_task=${tag} \
  --en_beta=0.0 \
  --focal_gamma=2.0 \
  --data_format=multi_files \
  --warmup_proportion=0.15 \
  --upsampling=false \
  --loss_mean_type=dark_mean \
  --gen_synthesis=true \
  --do_train=true \
  --do_eval=false \
  --do_train_and_eval=false \
  --eval_data=positive \
  --data_dir=/tmp/dummy \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=30.0 \
  --output_dir=${PROJECT_PATH}/model_data/multi_label_bert_classif
