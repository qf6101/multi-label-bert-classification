import os
from model_train.conf import BERT_PATH

# project path
pro_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# bert model params
config_file = str(BERT_PATH / 'bert_config.json')
vocab_file = str(BERT_PATH / 'vocab.txt')

# model params
model_dir = os.path.join(pro_path, "model_data/direct_bert_classif/")

biz1_model_file = os.path.join(model_dir, "biz1/model.ckpt-final")
biz2_model_file = os.path.join(model_dir, "biz2/model.ckpt-final")

merged_graph_file = os.path.join(model_dir, "merged/multi_label_classif_model.pb")

fp16 = False
xla = False
verbose = False
do_lower_case = True

from model_train.bert.biz1_args import input_files as biz1_input_files
from model_train.bert.biz2_args import input_files as biz2_input_files

n_class_biz1 = len(biz1_input_files)
n_class_biz2 = len(biz2_input_files)
