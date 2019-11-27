import os
import json
import logging
import contextlib
import collections
from termcolor import colored
from model_train.bert import modeling
import argparse


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

        if args.fp16:
            logger.warning("fp16 is turned on! "
                           "Note that not all CPU GPU support fast fp16 instructions, "
                           "worst case you will have degraded performance!")
        logger.info("model config: {}".format(args.config_file))
        logger.info(
            "checkpoint{}: {}".format(" (override by the fine-tuned model)" if args.model_dir else "",
                                      args.model_file))
        with tf.gfile.GFile(args.config_file, "r") as f:
            bert_config = modeling.BertConfig.from_dict(json.load(f))

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

            output_layer = model.get_pooled_output()
            hidden_size = output_layer.shape[-1].value

            if hasattr(args, 'n_class'):
                n_class = args.n_class
            else:
                n_class = len(args.input_files)
            output_weights = tf.get_variable(
                "bert/fc/output_weights", [n_class, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            output_bias = tf.get_variable(
                "bert/fc/output_bias", [n_class], initializer=tf.zeros_initializer())
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            init_vars = tf.train.list_variables(args.model_file)
            print(init_vars)
            assignment_map = collections.OrderedDict()
            for x in init_vars:
                (name, var) = (x[0], x[1])
                # if name not in name_to_variable:
                #     continue
                if 'adam' not in name.lower() and 'global_step' not in name.lower() and 'signal_early_stopping' not in name.lower():
                    assignment_map[name] = name
            print(assignment_map)
            tf.train.init_from_checkpoint(args.model_file, assignment_map)

            if args.fp16:
                logits = tf.cast(logits, tf.float16)

            logits = tf.identity(logits, "final_logits")
            output_tensors = [logits]
            tmp_g = tf.get_default_graph().as_graph_def()

        with tf.Session(config=config) as sess:
            logger.info("load parameters from checkpoint...")

            sess.run(tf.global_variables_initializer())
            dtypes = [n.dtype for n in input_tensors]
            logger.info("optimize...")
            tmp_g = optimize_for_inference(
                tmp_g,
                [n.name[:-2] for n in input_tensors],
                [n.name[:-2] for n in output_tensors],
                [dtype.as_datatype_enum for dtype in dtypes],
                False)

            logger.info("freeze...")
            tmp_g = convert_variables_to_constants(sess, tmp_g, [n.name[:-2] for n in output_tensors],
                                                   use_fp16=args.fp16)

        logger.info("write graph to file: {}".format(args.graph_file))
        with tf.gfile.GFile(args.graph_file, "wb") as f:
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
    parser = argparse.ArgumentParser(description='Make model pb file.')
    parser.add_argument('--biz_task', required=True, choices=['biz1', 'biz2'],
                        default='biz1', help='Business task type.')
    parser.add_argument('--certain_step', help='Certain step of model.', default=None)

    flags = parser.parse_args()

    print(flags)

    if flags.biz_task == 'biz1':
        from model_train.bert import biz1_args as args
    elif flags.biz_task == 'biz2':
        from model_train.bert import biz2_args as args
    else:
        raise ValueError('Wrong argument for making mode pb file.')

    from utils.pb_maker.multi_label_bert_classifier.pickup_model_file import pickup_model_file

    pickup_model_file(args, flags.certain_step)
    optimize_graph(args)
