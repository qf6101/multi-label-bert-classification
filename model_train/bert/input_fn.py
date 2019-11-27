import tensorflow as tf
from model_train.bert import tokenization


class InputExample(object):
    def __init__(self, unique_id, text_a, text_b=None, label=None):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, label_id=None):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.label_id = label_id

    def __str__(self):
        s = ""
        s += str(self.unique_id) + "\n"
        s += " ".join(self.tokens) + "\n"


def convert_sents_to_examples(sentences):
    assert isinstance(sentences, list)
    unique_id = 0
    for ss in sentences:
        line = tokenization.convert_to_unicode(ss)
        if not line:
            continue
        line = line.strip()
        text_a = line
        text_b = None
        yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
        unique_id += 1


def convert_sents_to_features(sents, seq_length, tokenizer):
    for (ex_index, example) in enumerate(convert_sents_to_examples(sents)):
        tokens_a = tokenizer.tokenize(example.text_a)

        # if the sentences's length is more than seq_length, only use sentence's left part
        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        # Where "input_ids" are tokens's index in vocabulary
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        yield {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "input_type_ids": input_type_ids
        }


def convert_examples_to_features(examples, seq_length, tokenizer):
    assert isinstance(examples, list)
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        # Where "input_ids" are tokens's index in vocabulary
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                label_id=example.label
            )
        )
    return features


def convert_example_to_feature(example, seq_length, tokenizer):
    tokens_a = tokenizer.tokenize(example.text_a)

    if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    # Where "input_ids" are tokens's index in vocabulary
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return InputFeatures(
        unique_id=example.unique_id,
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids,
        label_id=example.label
    )


def input_fn_builder(batch_size, max_seq_len, tokenizer, repeat=1):
    def input_fn(sents):
        dataset = tf.data.Dataset.from_generator(
            lambda: convert_sents_to_features(
                sents=sents, seq_length=max_seq_len, tokenizer=tokenizer
            ),
            output_types={
                "input_ids": tf.int32,
                "input_mask": tf.int32,
                "input_type_ids": tf.int32
            },
            output_shapes={
                "input_ids": (max_seq_len,),
                "input_mask": (max_seq_len,),
                "input_type_ids": (max_seq_len,)
            }
        )
        dataset = dataset.repeat(repeat)
        dataset = dataset.batch(batch_size=batch_size)
        data_iter = dataset.make_one_shot_iterator()
        batch = data_iter.get_next()
        return batch

    return input_fn
