import os
import tensorflow as tf
from model_train.conf import BERT_PATH

# compat tensorflow 2.0
if tuple(map(int, tf.__version__.split("."))) >= (2, 0, 0):
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

tf.logging.set_verbosity(tf.logging.INFO)

# project path
pro_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# data path
data_dir = os.path.join(pro_path, "data/biz1")

# bert model params
config_file = str(BERT_PATH / 'bert_config.json')
bert_file = str(BERT_PATH / 'bert_model.ckpt')
vocab_file = str(BERT_PATH / 'vocab.txt')

# freeze layers
freeze_layers = ["embeddings", "layer_0", "layer_1", "layer_2", "layer_3", "layer_4", "layer_5", "layer_6", "layer_7",
                 "layer_8", "layer_9", "layer_10"]

model_dir = os.path.join(pro_path, "model_data/multi_label_bert_classif/biz2/")
model_file = os.path.join(model_dir, "model.ckpt-final")
graph_file = os.path.join(model_dir, "multi_label_classif_model.pb")
save_model_path = os.path.join(model_dir, "model.ckpt")

input_files = [
    os.path.join(data_dir, "unknown.txt"),
    os.path.join(data_dir, "some1.txt"),
    os.path.join(data_dir, "some2.txt"),
    os.path.join(data_dir, "some3.txt"),
    os.path.join(data_dir, "some4.txt")
]

trans = {
    "unknown": "biz_name_unknown",
    "some1": "biz_name1",
    "some2": "biz_name2",
    "some3": "biz_name3",
    "some4": "biz_name4"
}

fp16 = False
xla = False
verbose = False
do_lower_case = True

inter_op_parallelism_threads = 0
intra_op_parallelism_threads = 0
gpu_allow_growth = True
log_device_placement = False
allow_soft_placement = False

synthesis_classes = [
    ('some1', 'some2'),
    ('some1', 'some3'),
    ('some2', 'some3'),
    ('some3', 'some4')
]

synthesis_size = 5

excluding = {
    'some1': ['some4'],
    'some2': ['some4'],
    'some4': ['some1', 'some2']
}
