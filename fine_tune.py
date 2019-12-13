#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File fine_tune  
@Time 2019/12/9 下午5:53
@Author wushib
@Description 迁移学习与微调

第一步，调用keras application里面的模型，去掉top分类层
第二步，冻结卷积基，添加自定义分类层或者池化层
第三步，训练10个批次，得到95%的准确率
第四步，在前三步的基础上，解冻部分层，重新编译模型，设置初始批次为10，继续训练10个批次，得到98%的准确率

note:微调是在迁移学习的基础上进行的，先冻结所有基础层，接着训练自定义分类层，然后解冻部分基础层，再次训练

MobileNetV2模型期望输出被标准化至 [-1,1] 范围内，所以在我们传入输入数据时，需要将数据标准化到[-1,1]

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras


tf.data.Dataset
IMG_SIZE = 160
BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 1000

SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
(raw_train, raw_validation, raw_test), metadata = tfds.load('cats_vs_dogs', split=list(splits), with_info=True,
                                                            as_supervised=True)
print(raw_train)
print(raw_validation)
print(raw_test)

get_label_name = metadata.features['label'].int2str


def format_example(image, label):
    image = tf.cast(image, tf.float32)  # float32为实数[0,255]
    image = (image / 127.5)-1 # 调整为[-1, 1]
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

for image_batch, label_batch in train_batches.take(1):
    pass

print(image_batch.shape)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

feature_batch = base_model(image_batch)
print(feature_batch.shape)

# 冻结卷积基
base_model.trainable = False
# base_model.summary()
global_average_layer = keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
prediction_layer = keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

num_train, num_val, num_test = (
    metadata.splits['train'].num_examples * weight / 10
    for weight in SPLIT_WEIGHTS
)
initial_epochs = 10
steps_per_epoch = round(num_train) // BATCH_SIZE
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

'''
loss: 0.4962 - accuracy: 0.9388 - val_loss: 0.6479 - val_accuracy: 0.9401
'''


# 解冻卷积基
base_model.trainable = True

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(train_batches,
                         epochs=total_epochs,
                         initial_epoch =  history.epoch[-1],
                         validation_data=validation_batches)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

"""
loss: 0.2584 - accuracy: 0.9830 - val_loss: 0.3296 - val_accuracy: 0.9634
"""


