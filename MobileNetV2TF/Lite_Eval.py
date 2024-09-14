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

    mobilenet_model = loaded_model.layers[0]
    classifier_head = loaded_model.layers[1]

    os.makedirs('./tmp', exist_ok=True)
    mobilenet_model.save_weights('./tmp/tmp_weights_mbl.weights.h5')

    model = MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=True,
        weights='./tmp/tmp_weights_mbl.weights.h5',
        input_tensor=None,
        pooling=None,
        classes=101,
        classifier_activation='softmax'
    )

    os.remove('./tmp/tmp_weights_mbl.weights.h5')
    os.removedirs('./tmp')

    out_model = keras.Sequential([
        model,
        classifier_head
    ])

    out_model.trainable = False  # Freeze the model

    return out_model

def convert_model_to_tflite(model: keras.Model):
    '''
    Convert a Keras model to TensorFlow Lite format and save it to a file.

    :param model: The Keras model to convert
    :param tflite_model_path: The path to save the converted TFLite model
    
    :return: The converted TFLite model
    '''
    converter = lite.TFLiteConverter.from_keras_model(model)
    lite_model = converter.convert()

    return lite_model

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def preprocess_mobilenet(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # Apply MobileNetV2-specific preprocessing
    return image, label

def resize_img(image, label):
    return tf.image.resize(image, (224, 224)), label

def one_hot_encode(image, label):
        label = tf.one_hot(label, depth=100)
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

if __name__ == '__main__':

    try:
        os.mkdir("Benchmarking")
    except:
        pass
    path = f"Benchmarking/{datetime.datetime.now().date()}_{str(datetime.datetime.now().time()).split('.')[0].replace(':', '-')}"
    os.mkdir(path)

    logging.basicConfig(level=logging.INFO, filename=f'{path}/logs.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Transfer Learning with MobileNetV2 on CIFAR-100')
    parser.add_argument('-b', '--batch', type=int, default=32, help='Batch size for training')
    parser.add_argument('-m', '--model', type=str, default="", help='File to load the model from')
    # parser.add_argument('-lm', '--lite', type=str, default=None, help='File to load the lite model from')
    args = parser.parse_args()

    logging.info("Arguments: %s", args)

    BATCH_SIZE = args.batch
    LR = 1e-3

    logging.info("Loading dataset Cifar100")
    datasets, ds_info = load_dataset('cifar100', ['train', 'test'])
    ds_train, ds_val, ds_test = datasets

    ds_train = ds_train.batch(BATCH_SIZE)
    ds_val = ds_val.batch(BATCH_SIZE)
    ds_test = ds_test.batch(BATCH_SIZE)

    logging.info("Loading model")
    model = None
    if args.model:
        model = load_model(args.model)
        model.summary()
    else:
        raise ValueError("No model file provided")

    print("Compiling model")
    logging.info("Compiling model")
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            'precision',
            'recall',
            'f1_score'
        ]
    )

    logging.info("Evaluating model")
    test_loss, test_acc, test_f1, test_precision, test_cm = model.evaluate(ds_test)

    logging.info("Test accuracy: %.4f", test_acc)
    logging.info("Test loss: %.4f", test_loss)
    logging.info("Test F1 score: %.4f", test_f1)
    logging.info("Test precision: %.4f", test_precision)

    # at this point, we can't run the code as of (9/13/24) because LiteRT isn't available on PyPI yet
    raise(Exception("LiteRT not available yet. Exiting."))

    logging.info("Converting model to TFLite format")
    print("\nConverting model to TFLite format\n")
    interpreter = Interpreter(model_content=convert_model_to_tflite(model))
    interpreter.allocate_tensors()

    """
    logging.info("Evaluating Lite model")
    test_loss, test_acc, test_f1, test_precision, test_cm = lite_model.evaluate(ds_test)

    logging.info("Lite Test accuracy: %.4f", test_acc)
    logging.info("Lite Test loss: %.4f", test_loss)
    logging.info("Lite Test F1 score: %.4f", test_f1)
    logging.info("Lite Test precision: %.4f", test_precision)
    """
