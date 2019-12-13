#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File tf_hub  
@Time 2019/12/11 下午3:54
@Author wushib
@Description None
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import tensorflow.keras as keras

model = keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False)

help(keras.applications.mobilenet_v2.preprocess_input)
