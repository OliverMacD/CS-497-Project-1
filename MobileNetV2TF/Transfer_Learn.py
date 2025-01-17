from typing import List, Tuple, Any

import os

import logging

import argparse

import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

import tensorflow_datasets as tfds

import numpy as np

import matplotlib.pyplot as plt


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def save_model(model, filepath: str = "", name: str | None = None,):
    '''
    Save the model to a file
    
    :param model: The model to serialize
    :param filepath: The path to save the model to
    :param name: The name of the file to save the model to

    :return: None

    See [reference](https://www.tensorflow.org/guide/keras/serialization_and_saving) for more information
    '''
    if name is None:
        name = ""
        today = str(datetime.datetime.now().date())
        time = str(datetime.datetime.now().time()).split(".")[0].replace(":", "-")
        name = f"model_{today}_{time}.keras"
    model.save(filepath + name)

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

def append_classifier(model: keras.Model) -> keras.Model:
    '''
    Append a layer to a model
    
    :param model: The model to append the layer to
    
    :return: The model with the layer appended
    '''
    new_model = keras.Sequential()
    new_model.add(model)

    prediction_layer = tf.keras.layers.Dense(100, activation='softmax')
    new_model.add(prediction_layer)

    return new_model


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

    return (ds_train, ds_val, ds_test), ds_info

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

def plot_training_metrics(
        history,
        show: bool = False,
        save: bool = False,
        path: str = "",
        name: str = "training_metrics.png"
    ):
    '''
    Plot training metrics

    :param history: The history object returned by model.fit()
    :param show: Whether to show the plot
    :param save: Whether to save the plot
    :param path: The path to save the plot to
    :param name: The name of the file to save the plot to

    :return: None
    '''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.title('Cifar100 Fine-Tuning Metrics')

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    if save:
        plt.savefig(os.path.join(path, name))
    if show:
        plt.show()


if __name__ == '__main__':

    try:
        os.mkdir("Training_Results")
    except:
        pass
    path = f"Training_Results/{datetime.datetime.now().date()}_{str(datetime.datetime.now().time()).split('.')[0].replace(':', '-')}"
    os.mkdir(path)

    logging.basicConfig(level=logging.INFO, filename=f'{path}/logs.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Transfer Learning with MobileNetV2 on CIFAR-100')
    parser.add_argument('-b', '--batch', type=int, default=32, help='Batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs to train for')
    parser.add_argument('-l', '--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('-f', '--file', type=str, default="", help='File to load the model from')
    parser.add_argument('-s', '--save', type=str, default=None, help='Name of the model file to save')
    args = parser.parse_args()

    logging.info("Arguments: %s", args)

    BATCH_SIZE = args.batch
    EPOCHS = args.epochs
    LR = args.lr

    logging.info("Loading dataset Cifar100")
    datasets, ds_info = load_dataset('cifar100', ['train', 'test'])
    ds_train, ds_val, ds_test = datasets

    logging.info("Preprocessing dataset")
    ds_train = ds_train.map(preprocess_mobilenet)
    ds_train = ds_train.map(resize_img)
    ds_val = ds_val.map(preprocess_mobilenet)
    ds_val = ds_val.map(resize_img)
    ds_test = ds_test.map(preprocess_mobilenet)
    ds_test = ds_test.map(resize_img)

    ds_train = ds_train.map(one_hot_encode)
    ds_val = ds_val.map(one_hot_encode)
    ds_test = ds_test.map(one_hot_encode)

    ds_train = ds_train.batch(BATCH_SIZE)
    ds_val = ds_val.batch(BATCH_SIZE)
    ds_test = ds_test.batch(BATCH_SIZE)

    logging.info("Building model")
    model = None
    if args.file:
        model = load_model(args.file)
        model.trainable = False
        model = append_classifier(model)
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

    print("Training Classifier")
    print(f'Start Time: {datetime.datetime.now()}\n')
    logging.info("Training classifier")
    history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS)

    plot_training_metrics(history, save=True, name="training_metrics_classification.png", path=path)

    test_loss, test_acc, test_f1, test_precision, test_cm = model.evaluate(ds_test)

    logging.info(f"Test accuracy: {test_acc}")
    logging.info(f"Test loss: {test_loss}")
    logging.info(f"Test F1 score: {test_f1}")
    logging.info(f"Test precision: {test_precision}")
    logging.info(f"Test confusion matrix saved to confusion_matrix.npy")
    np.save(os.path.join(path, "confusion_matrix.npy"), test_cm)
    print(f"Test accuracy: {test_acc}")
    print(f"Test loss: {test_loss}")

    print("Fine-tuning model")
    logging.info("Fine-tuning model")
    for layer in model.layers[0].layers:
        layer.trainable = True
    
    print("Compiling model")
    logging.info("Compiling model")
    model.compile(
        optimizer=Adam(learning_rate=LR/10),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            'precision',
            'recall',
            'f1_score'
        ]
    )

    print("Training model")
    logging.info("Training model")
    history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS)

    plot_training_metrics(history, save=True, name="training_metrics_fine_tuning.png", path=path)

    test_loss, test_acc, test_f1, test_precision, test_cm = model.evaluate(ds_test)

    logging.info(f"Test accuracy: {test_acc}")
    logging.info(f"Test loss: {test_loss}")
    logging.info(f"Test F1 score: {test_f1}")
    logging.info(f"Test precision: {test_precision}")
    logging.info(f"Test confusion matrix saved to confusion_matrix.npy")
    np.save(os.path.join(path, "confusion_matrix.npy"), test_cm)
    print(f"Test accuracy: {test_acc}")
    print(f"Test loss: {test_loss}")

    print("Saving model")
    logging.info("Saving model")
    if args.save:
        save_model(model, filepath=path, name=args.save)
    else:
        save_model(model, filepath=path)
    print("Model saved")
    logging.info("Model saved")
