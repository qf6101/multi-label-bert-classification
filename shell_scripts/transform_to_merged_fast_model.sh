#!/usr/bin/env bash

PROJECT_PATH=$(cd $(dirname $(cd $(dirname $0); pwd)); pwd)


python -m utils.fast_model_pb_maker.multi_label_bert_classifier.make_merged_model_pb
