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


if __name__ == "__main__":
    # TFLite is now LiteRT.
    # see [reference](https://ai.google.dev/edge/litert/models/convert_tf)
    
    try:
        os.mkdir("Lite_Models")
    except:
        pass
    path = f"Lite_Models/{datetime.datetime.now().date()}_{str(datetime.datetime.now().time()).split('.')[0].replace(':', '-')}"
    os.mkdir(path)

    logging.basicConfig(level=logging.INFO, filename=f'{path}/logs.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Transfer Learning with MobileNetV2 on CIFAR-100')
    parser.add_argument('-f', '--file', type=str, default="", help='File to load the model from')
    parser.add_argument('-s', '--save', type=str, default=None, help='Name of the model file to save')
    args = parser.parse_args()

    if not args.file:
        logging.error("No model file provided")
        raise ValueError("No model file provided")
    if not args.save:
        logging.error("No save path provided")
        raise ValueError("No save path provided")

    logging.info("Arguments: %s", args)

    logging.info("Loading model from %s", args.file)
    model = None
    if args.file:
        model = load_model(args.file)
        model.summary()
    
    logging.info("Converting model to TFLite format")
    print("\nConverting model to TFLite format\n")
    converter = lite.TFLiteConverter.from_keras_model(model)
    lite_model = converter.convert()

    if args.save:
        with open(os.path.join(path, args.save), 'wb') as f:
            f.write(lite_model)
            logging.info("Model saved to %s", args.save)