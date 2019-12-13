#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File predicting_movie_reviews_with_bert_on_tf_hub  
@Time 2019/12/13 下午5:18
@Author wushib
@Description 预测电影评论示例
"""

import tensorflow as tf
import tensorflow_hub as hub
import bert
from bert import tokenization
from bert import optimization
from bert import run_classifier
from tensorflow import keras
import os
import re
import pandas as pd

"""
加载数据
"""


def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


def download_and_load_datasets(force_download=False):
    """
    :param force_download:
    """
    dataset = keras.utils.get_file(fname="aclImdb.tar.gz",
                                   origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                                   extract=True)
    train_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))
    return train_df, test_df


train, test = download_and_load_datasets()

"""
数据预处理
"""

DATA_COLUMN = 'sentence'
LABEL_COLUMN = 'polarity'
# label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'
label_list = [0, 1]

train_InputExamples = train.apply(
    lambda x: bert.run_classifier.InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this example
                                               text_a=x[DATA_COLUMN],
                                               text_b=None,
                                               label=x[LABEL_COLUMN]), axis=1)

test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                           text_a=x[DATA_COLUMN],
                                                                           text_b=None,
                                                                           label=x[LABEL_COLUMN]), axis=1)

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


def create_tokenizer_from_hub_module():
    """
    :return:
    """
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    return bert.tokenization.FullTokenizer(vocab_file=tokenization_info["vocab_file"],
                                           do_lower_case=tokenization_info["do_lower_case"])


tokenizer = create_tokenizer_from_hub_module()

tokenizer.tokenize("This here's an example of using the BERT tokenizer")
