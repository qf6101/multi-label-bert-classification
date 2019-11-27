import os
import json
import logging
import contextlib
from termcolor import colored

from model_train.bert import modeling
from model_train.bert.bert_vars import bert_vars
from model_train.bert.modeling import (
    gelu, relu, get_shape_list, reshape_to_matrix, attention_layer, create_initializer,
    dropout, layer_norm, reshape_from_matrix, create_attention_mask_from_input_mask
)

vars = [
    # layer_11
    "bert/encoder/layer_11/attention/output/LayerNorm/beta",
    "bert/encoder/layer_11/attention/output/LayerNorm/gamma",
    "bert/encoder/layer_11/attention/output/dense/bias",
    "bert/encoder/layer_11/attention/output/dense/kernel",
    "bert/encoder/layer_11/attention/self/key/bias",
    "bert/encoder/layer_11/attention/self/key/kernel",
    "bert/encoder/layer_11/attention/self/query/bias",
    "bert/encoder/layer_11/attention/self/query/kernel",
    "bert/encoder/layer_11/attention/self/value/bias",
    "bert/encoder/layer_11/attention/self/value/kernel",
    "bert/encoder/layer_11/intermediate/dense/bias",
    "bert/encoder/layer_11/intermediate/dense/kernel",
    "bert/encoder/layer_11/output/LayerNorm/beta",
    "bert/encoder/layer_11/output/LayerNorm/gamma",
    "bert/encoder/layer_11/output/dense/bias",
    "bert/encoder/layer_11/output/dense/kernel",

    # dense
    "bert/pooler/dense/bias",
    "bert/pooler/dense/kernel",

    # fc
    'bert/fc/output_weights',
    'bert/fc/output_bias'
]

BERT_START_LAYER = 11


def transformer_model(input_tensor, tf,
                      attention_mask=None,
                      hidden_size=768,
                      start_layer_index=BERT_START_LAYER,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.0,
                      attention_probs_dropout_prob=0.0,
                      initializer_range=0.02,
                      do_return_all_layers=True):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".
    This is almost an exact implementation of the original Transformer encoder.
    See the original paper:
    https://arxiv.org/abs/1706.03762
    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
      do_return_all_layers: Whether to also return all layers or just the final
        layer.
    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.
    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(start_layer_index, num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                        attention_output,
                        hidden_size,
                        kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range))

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


def set_logger(context, verbose=False):
    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        fmt="%(levelname)-.1s:" + context + ":[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s",
        datefmt="%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


def import_tf(device_id=-1, verbose=False, use_fp16=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if device_id < 0 else str(device_id)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0" if verbose else "3"
    os.environ["TF_FP16_MATMUL_USE_FP32_COMPUTE"] = "0" if use_fp16 else "1"
    os.environ["TF_FP16_CONV_USE_FP32_COMPUTE"] = "0" if use_fp16 else "1"
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    return tf


def optimize_graph(args, logger=None):
    if not logger:
        logger = set_logger(colored("GRAPHOPT", "cyan"), args.verbose)
    try:
        # we don't need GPU for optimizing the graph
        tf = import_tf(verbose=args.verbose)
        from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

        config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)

        config_fp = args.config_file
        biz1_init_checkpoint = args.biz1_model_file
        biz2_init_checkpoint = args.biz2_model_file
        if args.fp16:
            logger.warning(
                "fp16 is turned on! "
                "Note that not all CPU GPU support fast fp16 instructions, "
                "worst case you will have degraded performance!"
            )
        logger.info("model config: {}".format(config_fp))
        logger.info(
            "checkpoint{}: {}, {}, {}, {}".format(
                " (override by the fine-tuned model)" if args.model_dir else "",
                biz1_init_checkpoint, biz2_init_checkpoint
            )
        )

        with tf.gfile.GFile(config_fp, "r") as f:
            json_data = json.load(f)
            bert_config = modeling.BertConfig.from_dict(json_data)
            if json_data["hidden_act"] == "relu":
                act_fn = relu
            elif json_data["hidden_act"] == "gelu":
                act_fn = gelu
            else:
                raise ValueError('Invalid hidden activation function')

        logger.info("build graph...")
        # input placeholders, not sure if they are friendly to XLA
        input_ids = tf.placeholder(tf.int32, (None, None), "input_ids")
        input_mask = tf.placeholder(tf.int32, (None, None), "input_mask")
        input_type_ids = tf.placeholder(tf.int32, (None, None), "input_type_ids")

        jit_scope = tf.contrib.compiler.jit.experimental_jit_scope if args.xla else contextlib.suppress

        with jit_scope():
            input_tensors = [input_ids, input_mask, input_type_ids]

            model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=input_type_ids,
                use_one_hot_embeddings=False,
                scope="bert"
            )

            attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)

            inp_tensor = model.get_all_encoder_layers()[BERT_START_LAYER-1]
            # 0, 1, 2, 3, 4, 5, 6, 7 | 8, 9, 10

            output_tensors = []
            for tag, n_class in zip(['biz1', 'biz2'],
                                    [args.n_class_biz1, args.n_class_biz2]):
                with tf.variable_scope(tag):
                    with tf.variable_scope("encoder"):
                        all_encoder_layers = transformer_model(
                            inp_tensor, tf, attention_mask,
                            intermediate_act_fn=act_fn
                        )
                        sequence_output = all_encoder_layers[-1]

                    with tf.variable_scope("pooler"):
                        # We "pool" the model by simply taking the hidden state corresponding
                        # to the first token. We assume that this has been pre-trained
                        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
                        pooled_output = tf.layers.dense(
                            first_token_tensor,
                            768,
                            activation=tf.tanh,
                            kernel_initializer=create_initializer(0.02))

                    with tf.variable_scope("fc"):
                        hidden_size = pooled_output.shape[-1].value
                        output_weights = tf.get_variable(
                            "output_weights", [n_class, hidden_size],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
                        output_bias = tf.get_variable(
                            "output_bias", [n_class], initializer=tf.zeros_initializer())
                        logits = tf.matmul(pooled_output, output_weights, transpose_b=True)
                        logits = tf.nn.bias_add(logits, output_bias)

                if args.fp16:
                    logits = tf.cast(logits, tf.float16)

                logits = tf.identity(logits, tag + '_final_logits')
                output_tensors.append(logits)

            tmp_g = tf.get_default_graph().as_graph_def()

        with tf.Session(config=config) as sess:
            logger.info("load parameters from checkpoint...")

            sess.run(tf.global_variables_initializer())

            reader_biz1 = tf.train.NewCheckpointReader(args.biz1_model_file)
            reader_biz2 = tf.train.NewCheckpointReader(args.biz2_model_file)

            base_assign_ops = []
            assign_ops = []

            for name in bert_vars:
                try:
                    tensor = sess.graph.get_tensor_by_name('{}:0'.format(name))
                    weight = reader_biz1.get_tensor(name)
                    base_assign_ops.append(tf.assign(tensor, weight))
                except KeyError as e:
                    print(e, name)
                    raise e
                except Exception as e:
                    print(e, name)
                    raise e

            for tag, reader in zip(['biz1', 'biz2'],
                                   [reader_biz1, reader_biz2]):
                for name in vars:
                    try:
                        tensor = sess.graph.get_tensor_by_name('{}:0'.format(name.replace("bert", tag)))
                        weight = reader.get_tensor(name)
                        assign_ops.append(tf.assign(tensor, weight))
                    except KeyError as e:
                        print(e, name)
                        raise e
                    except Exception as e:
                        print(e, name)
                        raise e

            sess.run(base_assign_ops)
            sess.run(assign_ops)

            dtypes = [n.dtype for n in input_tensors]
            logger.info("optimize...")
            tmp_g = optimize_for_inference(
                tmp_g,
                [n.name[:-2] for n in input_tensors],
                [n.name[:-2] for n in output_tensors],
                [dtype.as_datatype_enum for dtype in dtypes],
                False
            )

            logger.info("freeze...")
            tmp_g = convert_variables_to_constants(sess, tmp_g, [n.name[:-2] for n in output_tensors],
                                                   use_fp16=args.fp16)

        logger.info("write graph to file: {}".format(args.merged_graph_file))
        with tf.gfile.GFile(args.merged_graph_file, "wb") as f:
            f.write(tmp_g.SerializeToString())
    except Exception:
        logger.error("fail to optimize the graph!", exc_info=True)


def convert_variables_to_constants(sess,
                                   input_graph_def,
                                   output_node_names,
                                   variable_names_whitelist=None,
                                   variable_names_blacklist=None,
                                   use_fp16=False):
    from tensorflow.python.framework.graph_util_impl import extract_sub_graph
    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.framework import node_def_pb2
    from tensorflow.core.framework import attr_value_pb2
    from tensorflow.core.framework import types_pb2
    from tensorflow.python.framework import tensor_util

    def patch_dtype(input_node, field_name, output_node):
        if use_fp16 and (field_name in input_node.attr) and (input_node.attr[field_name].type == types_pb2.DT_FLOAT):
            output_node.attr[field_name].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_HALF))

    inference_graph = extract_sub_graph(input_graph_def, output_node_names)

    variable_names = []
    variable_dict_names = []
    for node in inference_graph.node:
        if node.op in ["Variable", "VariableV2", "VarHandleOp"]:
            variable_name = node.name
            if ((variable_names_whitelist is not None and
                 variable_name not in variable_names_whitelist) or
                    (variable_names_blacklist is not None and
                     variable_name in variable_names_blacklist)):
                continue
            variable_dict_names.append(variable_name)
            if node.op == "VarHandleOp":
                variable_names.append(variable_name + "/Read/ReadVariableOp:0")
            else:
                variable_names.append(variable_name + ":0")
    if variable_names:
        returned_variables = sess.run(variable_names)
    else:
        returned_variables = []
    found_variables = dict(zip(variable_dict_names, returned_variables))

    output_graph_def = graph_pb2.GraphDef()
    how_many_converted = 0
    for input_node in inference_graph.node:
        output_node = node_def_pb2.NodeDef()
        if input_node.name in found_variables:
            output_node.op = "Const"
            output_node.name = input_node.name
            dtype = input_node.attr["dtype"]
            data = found_variables[input_node.name]

            if use_fp16 and dtype.type == types_pb2.DT_FLOAT:
                output_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(data.astype("float16"),
                                                             dtype=types_pb2.DT_HALF,
                                                             shape=data.shape)))
            else:
                output_node.attr["dtype"].CopyFrom(dtype)
                output_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(data, dtype=dtype.type,
                                                         shape=data.shape)))
            how_many_converted += 1
        elif input_node.op == "ReadVariableOp" and (input_node.input[0] in found_variables):
            # placeholder nodes
            # print('- %s | %s ' % (input_node.name, input_node.attr["dtype"]))
            output_node.op = "Identity"
            output_node.name = input_node.name
            output_node.input.extend([input_node.input[0]])
            output_node.attr["T"].CopyFrom(input_node.attr["dtype"])
            if "_class" in input_node.attr:
                output_node.attr["_class"].CopyFrom(input_node.attr["_class"])
        else:
            # mostly op nodes
            output_node.CopyFrom(input_node)

        patch_dtype(input_node, "dtype", output_node)
        patch_dtype(input_node, "T", output_node)
        patch_dtype(input_node, "DstT", output_node)
        patch_dtype(input_node, "SrcT", output_node)
        patch_dtype(input_node, "Tparams", output_node)

        if use_fp16 and ("value" in output_node.attr) and (
                output_node.attr["value"].tensor.dtype == types_pb2.DT_FLOAT):
            # hard-coded value need to be converted as well
            output_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
                tensor=tensor_util.make_tensor_proto(
                    output_node.attr["value"].tensor.float_val[0],
                    dtype=types_pb2.DT_HALF)))

        output_graph_def.node.extend([output_node])

    output_graph_def.library.CopyFrom(inference_graph.library)
    return output_graph_def


if __name__ == "__main__":
    from utils.pb_maker.multi_label_bert_classifier import args

    optimize_graph(args)
