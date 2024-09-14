from typing import List, Tuple, Any

import os

import logging

import argparse

import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow import lite

import tensorflow_datasets as tfds

import numpy as np

import matplotlib.pyplot as plt


def load_model(filepath: str) -> keras.Model:
    '''
    Load a model from a file

    This is a workaround for transfer learning not working in the latest stable version of TensorFlow.
    See [reference](https://github.com/keras-team/keras/issues/20188) for more information. I hate this.
    
    :param filepath: The path to the model file
    
    :return: The model
    '''
    loaded_model = keras.models.load_model(filepath)

    os.makedirs('./tmp', exist_ok=True)
    loaded_model.save_weights('./tmp/tmp_weights.weights.h5')

    model = MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=True,
        weights='./tmp/tmp_weights.weights.h5',
        input_tensor=None,
        pooling=None,
        classes=101,
        classifier_activation='softmax'
    )

    os.remove('./tmp/tmp_weights.weights.h5')
    os.removedirs('./tmp')

    return model

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def preprocess_mobilenet(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # Apply MobileNetV2-specific preprocessing
    return image, label

def resize_img(image, label):
    return tf.image.resize(image, (224, 224)), label

def one_hot_encode(image, label):
        label = tf.one_hot(label, depth=101)
        return image, label

def load_dataset(dataset_name: str, split: Tuple[str], split_perc: float = 0.8) -> Tuple[Tuple, Any]:
    '''
    Load a dataset from TensorFlow Datasets

    :param dataset_name: The name of the dataset to load
    :param split: The split to load
    :param split_perc: The percentage of the dataset to use for training

    :return: The dataset and dataset info
    '''
    (ds_train_val, ds_test), ds_info = tfds.load(
        dataset_name,
        split=split,
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    train_size = int(len(ds_train_val)*split_perc)

    # ds_train_val = ds_train_val.shuffle(buffer_size=min(shuffle_buffer_size, len(ds_train_val)))
    ds_train = ds_train_val.take(train_size)
    ds_train = ds_train.shuffle(buffer_size=train_size)
    ds_val = ds_train_val.skip(train_size)

    ds_train = ds_train.map(preprocess_mobilenet)
    ds_train = ds_train.map(resize_img)
    ds_val = ds_val.map(preprocess_mobilenet)
    ds_val = ds_val.map(resize_img)
    ds_test = ds_test.map(preprocess_mobilenet)
    ds_test = ds_test.map(resize_img)

    ds_train = ds_train.map(one_hot_encode)
    ds_val = ds_val.map(one_hot_encode)
    ds_test = ds_test.map(one_hot_encode)

    return (ds_train, ds_val, ds_test), ds_info

model = load_model('../../../Training_Results/2024-09-13_11-26-10/mobilenetv2_model.keras')  # Load the model

model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            'precision',
            'recall',
            'f1_score'
        ]
    )

# Load dataset
datasets, ds_info = load_dataset('food101', ['train', 'validation'])
ds_train, ds_val, ds_test = datasets

ds_train = ds_train.batch(32)
ds_val = ds_val.batch(32)
ds_test = ds_test.batch(32)

# evaluate the model
results = model.evaluate(ds_test)

# Print the results
print(len(results))
print(f'Test Loss: {results[0]:.4f}')
print(f'Test Accuracy: {results[1]:.4f}')
print(f'Test Precision: {results[2]:.4f}')
print(f'Test Recall: {results[3]:.4f}')
F1 = 2 * (results[2] * results[3]) / (results[2] + results[3])
print(f'Test F1 Score: {F1:.4f}')
