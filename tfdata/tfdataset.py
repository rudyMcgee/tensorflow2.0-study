#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File tfdataset  
@Time 2019/12/11 下午5:47
@Author wushib
@Description None
"""

import tensorflow as tf
import tensorflow_datasets as tfds

# 列出所有dataset
# print(tfds.list_builders())

ds_mnist = tfds.load(name="mnist", split="train", shuffle_files=True)
# 调用顺序shuffle、batch、repeat、prefetch
ds_mnist = ds_mnist.shuffle(1000).batch(1).repeat(5).prefetch(10)
for fe in ds_mnist.take(1):
    print(fe['image'], fe['label'])
