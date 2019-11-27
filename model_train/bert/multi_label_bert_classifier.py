# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
from model_train.bert import modeling
from model_train.bert import optimization
from model_train.bert import tokenization
import numpy as np
import tensorflow as tf
import random
from pathlib import Path
from sklearn.utils import resample, shuffle
from collections import defaultdict
import jieba.posseg as pseg

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "biz_task", None,
    "Business task type: biz1 and biz2. ")

flags.DEFINE_bool(
    "upsampling", None,
    "Upsample for imbalanced data. ")

flags.DEFINE_bool(
    "gen_synthesis", None,
    "Generate synthesis data. ")

flags.DEFINE_string(
    "data_format", None,
    "Data format. ")

flags.DEFINE_string(
    "loss_mean_type", None,
    "Loss mean type. ")

flags.DEFINE_string(
    "eval_data", None,
    "Evaluate data: all, positive or negative. ")

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_float(
    "en_beta", 0.0,
    "Beta parameter of effect number of samples")

flags.DEFINE_float(
    "focal_gamma", 0.0,
    "Gamma of focal loss")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_train_and_eval", False, "Whether to run train and eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class EvalCallBackFn(tf.train.SessionRunHook):
    def __init__(self):
        self.total_loss = 0
        self.n_example = 0

    def before_run(self, run_context):
        graph = tf.get_default_graph()
        self.loss_op = graph.get_operation_by_name("per_example_loss")
        self.element = self.loss_op.outputs[0]
        return tf.train.SessionRunArgs([self.element])

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        self.total_loss += run_values.results[0].sum()
        self.n_example += run_values.results[0].shape[0]
        # print(self.total_loss, self.n_example)

    def end(self, session):
        print("average loss: ", self.total_loss / self.n_example)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label_names=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label_names: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label_names = label_names


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_class_weights(self):
        raise NotImplementedError()

    def get_excluding(self):
        raise NotImplementedError()

    @classmethod
    def read_examples(cls, input_files):
        examples = []
        guid = 0
        label_names_of_examples = []

        for f in input_files:
            label_names = Path(f).stem.split('#')
            with open(f, "r", encoding="utf-8") as reader:
                while True:
                    line = reader.readline()
                    if not line:
                        break
                    line = line.strip().strip("\n")
                    if line == "":
                        line = " "

                    examples.append(InputExample(guid=guid, text_a=line, text_b=None, label_names=label_names))
                    label_names_of_examples.append(label_names)
                    guid += 1

        return examples, label_names_of_examples

    @classmethod
    def read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def replace_entity(cls, inp):
        outp = []
        for w in pseg.cut(inp):
            if w.flag == 'ns':
                outp.append('谶')
            else:
                outp.append(w.word)

        return ''.join(outp)

    @classmethod
    def create_examples(cls, lines, set_type):
        """Create examples and label_names for the training and dev sets."""
        examples = []
        label_names_of_examples = []

        for (i, line) in enumerate(lines):
            guid = i
            text_a = line[0].strip().strip("\n")
            text_a = cls.replace_entity(text_a)
            if text_a == "":
                text_a = " "
            if set_type == "test":
                label_names = "0"
            else:
                label_names = line[1].split("@@@")
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label_names=label_names))
            label_names_of_examples.append(label_names)
        return examples, label_names_of_examples

    @classmethod
    def gen_synthesis_data(cls, examples, synthesis_classes, synthesis_size):
        if len(synthesis_classes) == 0:
            return []

        useful_classes = set()
        for item in synthesis_classes:
            useful_classes.add(item[0])
            useful_classes.add(item[1])

        examples_by_labels = {}
        for ex in examples:
            for label in ex.label_names:
                if label in useful_classes and len(ex.text_a) <= synthesis_size:
                    examples_by_labels.setdefault(label, []).append(ex.text_a)

        synthesis_data = []
        counter = 10000000

        for item in synthesis_classes:
            if item[0] not in examples_by_labels or item[1] not in examples_by_labels:
                continue

            lhs = examples_by_labels[item[0]]
            rhs = examples_by_labels[item[1]]

            for m in lhs:
                for n in rhs:
                    if np.random.randint(2) == 0:
                        line = m + '，' + n
                    else:
                        line = n + '，' + m

                    synthesis_data.append(InputExample(guid=counter, text_a=line, text_b=None,
                                                       label_names=[item[0], item[1]]))
                    counter = counter + 1

        for i in range(5):
            random.shuffle(synthesis_data)

        return synthesis_data


class MyBusinessProcessor(DataProcessor):
    def __init__(self):
        if FLAGS.biz_task == 'biz1':
            from model_train.bert import biz1_args as args
        elif FLAGS.biz_task == 'biz2':
            from model_train.bert import biz2_args as args
        else:
            raise ValueError('Parameter biz_task error in data processing. ')

        tf.logging.info('biz task: {}'.format(FLAGS.biz_task))

        self.inp_files = args.input_files
        if FLAGS.data_format == 'multi_files':
            examples, label_names_of_examples = self.read_examples(args.input_files)
        elif FLAGS.data_format == 'one_file':
            examples, label_names_of_examples = self.create_examples(
                self.read_tsv(args.input_files[0]), "train")
        else:
            raise ValueError('Parameter data_format error in data processing. ')

        if FLAGS.gen_synthesis:
            synthesis_data = self.gen_synthesis_data(examples, args.synthesis_classes, args.synthesis_size)
            examples.extend(synthesis_data)

        for i in range(5):
            random.shuffle(examples)
        self.examples = examples

        label_names = [i for c in self.examples for i in c.label_names]

        if all(i.isnumeric() for i in label_names):
            self.classes = sorted(list(set([int(i) for i in label_names])))
            self.classes = [str(i) for i in self.classes]
            tf.logging.info('All labels are numeric. ')
        else:
            self.classes = sorted(list(set(label_names)))
            tf.logging.info('Not all labels are numeric. ')

        label_map = {}
        for (i, label) in enumerate(self.classes):
            label_map[label] = i

        if len(args.excluding) > 0:
            self.excluding = np.zeros(shape=[len(self.classes), len(self.classes)], dtype=np.int32)
            for k, v in args.excluding.items():
                ridx = label_map[k]
                for vi in v:
                    cidx = label_map[vi]
                    self.excluding[ridx, cidx] = 1
        else:
            self.excluding = None

        if self.excluding is not None:
            tf.logging.info('excluding: {}'.format(self.excluding.nonzero()))
        tf.logging.info('label map: {}'.format(label_map))

        if FLAGS.do_train_and_eval:
            classes = set()
            label_bins = defaultdict(list)
            for i in range(len(self.examples)):
                label = "_".join(sorted(self.examples[i].label_names))
                classes.add(label)
                label_bins[label].append(self.examples[i])
            label_bins_len = {c: len(label_bins[c]) for c in classes}
            self.trn_examples = []
            self.dev_examples = []

            rate = 0.1
            for c in classes:
                n_sample = int(label_bins_len[c] * (1 - rate))
                self.trn_examples.extend(label_bins[c][:n_sample])
                self.dev_examples.extend(label_bins[c][n_sample:])

            random.shuffle(self.trn_examples)
            random.shuffle(self.dev_examples)
        else:
            self.trn_examples = examples

        # reload label names
        label_names = np.array([i for c in self.trn_examples for i in c.label_names])

        if FLAGS.en_beta == 0.0:
            n_total_scaling = len(label_names) / len(self.classes)
            self.class_weights = np.array([n_total_scaling / len(label_names[label_names == k]) for k in self.classes])
        else:
            self.class_weights = np.array([1000 *
                                           (1. - FLAGS.en_beta) / (1. - np.float_power(FLAGS.en_beta, len(
                label_names[label_names == k])))
                                           for k in self.classes])
            self.class_weights = self.class_weights / np.sum(self.class_weights) * len(self.classes)

        tf.logging.info('en_beta: {}, class weights: {}'.format(FLAGS.en_beta, self.class_weights))

    def get_train_examples(self, data_dir):
        if FLAGS.upsampling:
            label_bins = defaultdict(list)
            classes = set()
            for i in range(len(self.trn_examples)):
                label = "_".join(sorted(self.examples[i].label_names))
                label_bins[label].append(self.trn_examples[i])
                classes.add(label)

            label_bins_len = {c: len(label_bins[c]) for c in classes}
            n_resample = max(label_bins_len.values())
            trn_examples = []
            for label in classes:
                if '_' in label:
                    resample_times = 2
                else:
                    resample_times = 5

                trn_examples.append(
                    resample(np.array(label_bins[label]),
                             n_samples=min(n_resample, len(label_bins[label]) * resample_times))

                )
            trn_examples = np.concatenate(trn_examples)
            return shuffle(trn_examples)
        else:
            return self.trn_examples

    def get_dev_examples(self, data_dir):
        if FLAGS.do_train_and_eval:
            return self.dev_examples
        else:
            if FLAGS.eval_data == 'positive':
                return [ex for ex in self.trn_examples if '0' in ex.label_names]
            elif FLAGS.eval_data == 'negative':
                return [ex for ex in self.trn_examples if '0' not in ex.label_names]
            else:
                return self.trn_examples

    def get_test_examples(self, data_dir):
        if FLAGS.data_format == "multi_files":
            examples, _ = self.read_examples([os.path.join(data_dir, 'test_input.txt')])
        elif FLAGS.data_format == "one_file":
            examples, _ = self.create_examples(
                self.read_tsv(os.path.join(data_dir, 'test_input.tsv')), "test")
        else:
            raise ValueError('Parameter data_format error in data processing. ')

        for ex in examples:
            ex.label_names = ['0']
        return examples

    def get_labels(self):
        return self.classes

    def get_class_weights(self):
        return self.class_weights

    def get_excluding(self):
        return self.excluding


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=[0],
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_ids = [label_map[i] for i in example.label_names]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s" % " ".join(example.label_names))
        tf.logging.info("label ids: %s" % " ".join([str(i) for i in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.VarLenFeature(tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 label_ids, num_labels, use_one_hot_embeddings, class_weights, excluding):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert"
    )

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    with tf.variable_scope("bert/fc"):
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.sigmoid(logits)

        indices = tf.stack([label_ids.indices[:, 0], tf.cast(label_ids.values, tf.int64)], axis=1)
        updates = tf.ones(tf.shape(label_ids.values)[0])
        one_hot_labels_4positive = tf.scatter_nd(indices, updates, [label_ids.dense_shape[0], num_labels])
        one_hot_labels_4negative = 1 - one_hot_labels_4positive

        focal = tf.where(one_hot_labels_4positive > 0,
                         1 - probabilities,
                         probabilities)

        focal = tf.pow(focal, FLAGS.focal_gamma)

        if excluding is not None:
            excluding_ = tf.constant(excluding, dtype=tf.float32)
            one_hot_labels_4excluding = tf.matmul(one_hot_labels_4positive, excluding_)
            one_hot_labels_4excluding = tf.where(one_hot_labels_4excluding > 0,
                                                 tf.fill(tf.shape(one_hot_labels_4excluding), 1.0),
                                                 tf.fill(tf.shape(one_hot_labels_4excluding), 0.0))

            one_hot_labels_4negative = one_hot_labels_4negative - one_hot_labels_4excluding

        if class_weights is not None:
            class_weights = tf.constant(class_weights, dtype=tf.float32)
            class_weights = class_weights[tf.newaxis, :]
            class_weights_negative = (num_labels - 1) / (num_labels - 1 / class_weights)
        else:
            class_weights, class_weights_negative = 1.0, 1.0

        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels_4positive,
                                                                   logits=logits, name="xentropy")

        per_example_loss = focal * per_example_loss

        label_counts_4positive = tf.reduce_sum(one_hot_labels_4positive, axis=-1)
        label_counts_4negative = tf.reduce_sum(one_hot_labels_4negative, axis=-1)

        if FLAGS.loss_mean_type != 'normal_mean':
            per_example_loss_4positive = tf.reduce_sum(per_example_loss * one_hot_labels_4positive * class_weights,
                                                       axis=-1) / label_counts_4positive

        if excluding is not None:
            label_counts_4excluding = tf.reduce_sum(one_hot_labels_4excluding, axis=-1)
            label_counts_4excluding_ = tf.where(tf.equal(label_counts_4excluding, 0.0),
                                                tf.fill(tf.shape(label_counts_4excluding), 1.0),
                                                label_counts_4excluding)
            per_example_loss_4excluding_sum = tf.reduce_sum(
                per_example_loss * one_hot_labels_4excluding * class_weights, axis=-1)
            per_example_loss_4excluding_sum_ = tf.where(tf.equal(label_counts_4excluding, 0.0),
                                                        tf.fill(tf.shape(label_counts_4excluding), 0.0),
                                                        per_example_loss_4excluding_sum)
            per_example_loss_4excluding = per_example_loss_4excluding_sum_ / label_counts_4excluding_

        if FLAGS.loss_mean_type == 'normal_mean':
            per_example_loss = tf.reduce_mean(per_example_loss * class_weights, axis=-1)
            tf.logging.info('class weight: {}'.format(class_weights))
            tf.logging.info('loss mean type: normal mean.')
        elif FLAGS.loss_mean_type == 'light_mean':
            per_example_loss_4negative = tf.reduce_sum(
                per_example_loss * one_hot_labels_4negative * class_weights, axis=-1) / label_counts_4negative
            per_example_loss = tf.add(per_example_loss_4positive, per_example_loss_4negative)

            if excluding is not None:
                per_example_loss = tf.add(per_example_loss, per_example_loss_4excluding)
                per_example_loss = per_example_loss / 3
            else:
                per_example_loss = per_example_loss / 2

            tf.logging.info('class weight: {}'.format(class_weights))
            tf.logging.info('loss mean type: light mean.')
        elif class_weights != 1.0 and FLAGS.loss_mean_type == 'dark_mean':
            sample_weights_negative = tf.reduce_sum(one_hot_labels_4positive * class_weights_negative,
                                                    axis=-1) / label_counts_4positive
            sample_weights_negative = sample_weights_negative[:, tf.newaxis]
            per_example_loss_4negative = tf.reduce_sum(
                per_example_loss * one_hot_labels_4negative * sample_weights_negative,
                axis=-1) / label_counts_4negative
            per_example_loss = tf.add(per_example_loss_4positive, per_example_loss_4negative)

            if excluding is not None:
                per_example_loss = tf.add(per_example_loss, per_example_loss_4excluding)
                per_example_loss = per_example_loss / 3
            else:
                per_example_loss = per_example_loss / 2

            tf.logging.info('class weight: {}'.format(class_weights))
            tf.logging.info('loss mean type: dark mean.')
        elif class_weights != 1.0 and FLAGS.loss_mean_type == 'deep_dark_mean' and excluding is not None:
            sample_weights_negative = tf.reduce_sum(one_hot_labels_4positive * class_weights_negative,
                                                    axis=-1) / label_counts_4positive
            sample_weights_negative = sample_weights_negative[:, tf.newaxis]
            per_example_loss_4negative = tf.reduce_sum(
                per_example_loss * one_hot_labels_4negative * sample_weights_negative,
                axis=-1) / label_counts_4negative

            per_example_loss_4excluding_v2 = tf.reduce_sum(
                per_example_loss * one_hot_labels_4excluding * sample_weights_negative,
                axis=-1) / label_counts_4excluding

            per_example_loss = tf.add(per_example_loss_4positive, per_example_loss_4negative)
            per_example_loss = tf.add(per_example_loss, per_example_loss_4excluding_v2)
            per_example_loss = per_example_loss / 3

            tf.logging.info('class weight: {}'.format(class_weights))
            tf.logging.info('loss mean type: deep dark mean.')
        else:
            raise ValueError('Wrong loss config.')

        loss = tf.reduce_mean(per_example_loss)

        return loss, per_example_loss, logits, probabilities


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, class_weights, excluding):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids[0]), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings, class_weights, excluding)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            if FLAGS.biz_task == 'biz1':
                from model_train.bert import biz1_args as args
            elif FLAGS.biz_task == 'biz2':
                from model_train.bert import biz2_args as args
            else:
                raise ValueError('Parameter biz_task error in data processing. ')

            train_op, lr, global_step = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, args.freeze_layers)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(total_loss, label_ids, probabilities, is_real_example):
                predictions = tf.where(probabilities > 0.9, tf.fill(tf.shape(probabilities), 1),
                                       tf.fill(tf.shape(probabilities), 0))
                predictions = tf.cast(predictions, tf.int32)

                indices = tf.stack([label_ids.indices[:, 0], tf.cast(label_ids.values, tf.int64)], axis=1)
                updates = tf.ones(tf.shape(label_ids.values)[0])
                one_hot_labels = tf.scatter_nd(indices, updates, [label_ids.dense_shape[0], num_labels])

                accuracy = tf.metrics.accuracy(
                    labels=one_hot_labels, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=total_loss)
                p = tf.metrics.precision(labels=one_hot_labels, predictions=predictions, weights=is_real_example)
                r = tf.metrics.recall(labels=one_hot_labels, predictions=predictions, weights=is_real_example)
                tf.summary.scalar('eval_accuracy', accuracy)
                tf.summary.scalar('eval_loss', loss)
                tf.summary.scalar('eval_precision', p)
                tf.summary.scalar('eval_recall', r)
                copy_per_example_loss = tf.identity(loss, name="per_example_loss")
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                    "eval_precision": p,
                    "eval_recall": r,
                }

            eval_metrics = (metric_fn,
                            [total_loss, label_ids, probabilities, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "my_business": MyBusinessProcessor
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if FLAGS.biz_task not in {'biz1', 'biz2'}:
        raise ValueError(
            'Parameter biz_task must be in `biz1` and `biz2`.'
        )

    if FLAGS.eval_data not in {'all', 'positive', 'negative'}:
        raise ValueError('Parameter eval_data must be in `all`, `positive`, `negative`.')

    if FLAGS.data_format not in {'multi_files', 'one_file'}:
        raise ValueError('Parameter data_format must be in `multi_files` and `one_file`')

    if (FLAGS.do_train and FLAGS.do_train_and_eval) or (FLAGS.do_eval and FLAGS.do_train_and_eval):
        raise ValueError('Parameter `do_train_and_eval` excludes `do_train` and `do_eval`.')

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_train_and_eval:
        raise ValueError(
            "At least one of `do_train`, `do_eval`, `do_train_and_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    FLAGS.output_dir = os.path.join(FLAGS.output_dir, FLAGS.biz_task)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    if FLAGS.upsampling:
        class_weights = None
    else:
        class_weights = processor.get_class_weights()

    excluding = processor.excluding

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=20,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host
        ))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train or FLAGS.do_train_and_eval:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        class_weights=class_weights,
        excluding=excluding)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        assert FLAGS.do_train_and_eval is False

        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        assert FLAGS.do_train_and_eval is False

        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_train_and_eval:
        assert FLAGS.do_train is False and FLAGS.do_eval is False

        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)

        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        from tensorflow.python.estimator.training import TrainSpec
        from tensorflow.python.estimator.training import EvalSpec
        from tensorflow.python.estimator.training import train_and_evaluate

        early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator,
            metric_name='eval_loss',
            max_steps_without_decrease=500,
            min_steps=1000,
            run_every_secs=10)

        train_spec = TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[early_stopping])
        eval_spec = EvalSpec(input_fn=eval_input_fn, steps=eval_steps, throttle_secs=1)

        train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.data_dir, "test_output.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    flags.mark_flag_as_required("biz_task")
    flags.mark_flag_as_required("upsampling")
    flags.mark_flag_as_required("data_format")
    flags.mark_flag_as_required("loss_mean_type")
    flags.mark_flag_as_required("gen_synthesis")
    flags.mark_flag_as_required("eval_data")
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
