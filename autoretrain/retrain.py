# -*- coding: utf-8 -*-

## Models instantiated dyanamically, don't delete these imports!!!
from keras.applications import VGG16, VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201, preprocess_input
## ---------------------------------------------------------------

import argparse
import os
import time
import pickle
import json
import keras
import cv2
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np

from sklearn.utils import shuffle
from PIL import Image, ImageEnhance

from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from calculate_statistics import plot_roc, plot_confusion_matrix, save_classification_report


_DEFAULT_DATASETS_DIR = "dataset"

_DEFAULT_LEARNING_RATE = 1e-5
_DEFAULT_BATCH_SIZE = 10

_TRAIN_SUBDIR = "train"
_TEST_SUBDIR = "test"

_SPLIT_TRAIN = "train"
_SPLIT_TEST = "test"

_MODELS_DIR = "models"

_TEST_PRED_FILE_NAME = "_test.pckl"


def create_model():
    global image_size

    print("Loading model: " + baseline_model_path)
    model = load_model(baseline_model_path)
    print("Model loaded.")

    print("Model instance: " + str(model))

    image_size = int(model.input.shape[1])
    print("Image size: " + str(image_size))

    print(model.summary())

    model_num_classes = model.layers[-1].output_shape[1]
    print("Model number of classes: " + str(model_num_classes))

    if model_num_classes != num_classes:
        print("FATAL ERROR: model number of classes is not equal to the number of classes in the training set")
        exit(-1)

    return model


def load_split_images(split_name):
    assert split_name in [_SPLIT_TRAIN, _SPLIT_TEST]

    categories = class_names
    dataset_split_dir = os.path.join(dataset_dir, split_name)

    X, y = [], []

    if os.path.isdir(dataset_split_dir):
        i = 0
        for category in categories:
            category_dir = os.path.join(dataset_split_dir, category)
            for myfile in os.listdir(category_dir):
                img = Image.open(os.path.join(category_dir, myfile))
                img = img.resize((image_size, image_size))

                X.append(np.asarray(img))
                y.append(i)
            i += 1

    return X, y


def load_images():
    X_train, y_train = load_split_images(_SPLIT_TRAIN)
    X_test, y_test = load_split_images(_SPLIT_TEST)

    X_train, y_train = shuffle(X_train, y_train)

    if K.image_data_format() == 'channels_first':
        X_train = np.array(X_train).reshape(np.array(X_train).shape[0], 3, image_size, image_size)
        X_test = np.array(X_test).reshape(np.array(X_test).shape[0], 3, image_size, image_size)
        input_shape = (3, image_size, image_size)
    else:
        X_train = np.array(X_train).reshape(np.array(X_train).shape[0], image_size, image_size, 3)
        X_test = np.array(X_test).reshape(np.array(X_test).shape[0], image_size, image_size, 3)
        input_shape = (image_size, image_size, 3)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test


def train_model(data, model, save_weights_path):
    X_train, y_train, X_test, y_test = data

    train_datagen = ImageDataGenerator(
        rotation_range=180, horizontal_flip=True, vertical_flip=True, shear_range=0.2, zoom_range=0.2, fill_mode='constant')
    test_datagen = ImageDataGenerator()

    train_datagen.fit(X_train)
    test_datagen.fit(X_test)

    history = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size), epochs=epochs, verbose=1)
    model.save(save_weights_path)

    retrains_log = open(os.path.join(_MODELS_DIR, "retrains.log"), "a+")

    score = model.evaluate_generator(test_datagen.flow(X_test, y_test, shuffle=False))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    retrains_log.write(
        str(timestamp) + "\t" + str(id) + "\t" + str(image_size) + "\t" + str(
            learning_rate) + "\t" + str(score[1]) + "\n")

    return history


def test_model(X, model, file_name):
    if len(X) > 0:
        y = model.predict(X)

        f = open(os.path.join(target_dir, file_name), 'wb')
        pickle.dump(y, f)
        f.close()

        return y
    return []


def do_transfer_learning(model, num_frozen_layers, optimizer):
    i = 0
    for layer in model.layers:
        i += 1
        if i <= num_frozen_layers:
            layer.trainable = False
        else:
            layer.trainable = True

        print('{0}:\t{1}'.format(layer.trainable, layer.name))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    X_train, y_train, X_test, y_test = load_images()

    save_weights_path = os.path.join(target_dir, train_result_dir_name + ".model")
    history = train_model(data=(X_train, y_train, X_test, y_test), model=model,
                          save_weights_path=save_weights_path)

    y_pred = test_model(X_test, model, _TEST_PRED_FILE_NAME)

    plot_roc(os.path.join(_MODELS_DIR, id, id + "_ROC.png"), class_names, y_test, y_pred, id + " ROC")
    plot_confusion_matrix(os.path.join(_MODELS_DIR, id, id + "_CM.png"), class_names, y_test, y_pred, normalize=False, title=id)
    plot_confusion_matrix(os.path.join(_MODELS_DIR, id, id + "_CM_Norm.png"), class_names, y_test, y_pred, normalize=True, title="Normalized " + id)
    save_classification_report(os.path.join(_MODELS_DIR, id, id + "_stats.json"), y_test, y_pred, class_names)


def learn(model):
    num_frozen_layers = 0 #1
    optimizer = Adam(lr=learning_rate)

    do_transfer_learning(model, num_frozen_layers=num_frozen_layers, optimizer=optimizer)


def run():
    model = create_model()
    learn(model)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=_DEFAULT_DATASETS_DIR,
        help="Root directory of the training, and test datasets.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=_DEFAULT_LEARNING_RATE,
        help="Learning rate.")
    parser.add_argument(
        "--baseline_model_path",
        type=str,
        help="'Path to the baseline model.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help="Batch size.")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs.")
    parser.add_argument(
        "--loss",
        type=str,
        default="categorical_crossentropy",
        help="Loss function.")
    parser.add_argument(
        "id", type=str,
        help="Unique identifier.")

    flags, unparsed = parser.parse_known_args()

    dataset_dir = flags.dataset_dir
    baseline_model_path = flags.baseline_model_path
    learning_rate = flags.learning_rate
    batch_size = flags.batch_size
    dropout = flags.dropout
    epochs = flags.epochs
    loss = flags.loss

    id = flags.id
    timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())

    print("--------------------------------------------------")
    print("ID: " + str(id))

    _TEST_PRED_FILE_NAME = id + _TEST_PRED_FILE_NAME

    arg_values = ""
    for arg in vars(flags):
        print(arg + "='" + str(getattr(flags, arg)) + "'")
    print("--------------------------------------------------")
    train_result_dir_name = id # time.strftime("%Y%m%d%H%M%S", time.gmtime()) + "," + arg_values
    target_dir = os.path.join(_MODELS_DIR, train_result_dir_name)
    print("target_dir --> " + target_dir)
    print("--------------------------------------------------")
    train_dir = os.path.join(dataset_dir, _TRAIN_SUBDIR)
    test_dir = os.path.join(dataset_dir, _TEST_SUBDIR)

    class_names = []
    for file in os.listdir(train_dir):
        if os.path.isdir(os.path.join(train_dir, file)):
            class_names.append(file)
    class_names.sort()

    test_class_names = []
    for file in os.listdir(test_dir):
        if os.path.isdir(os.path.join(test_dir, file)):
            test_class_names.append(file)
    test_class_names.sort()

    num_classes = len(class_names)
    print("train_dir --> " + train_dir)
    print("num_classes --> " + str(num_classes))
    print("classes_names --> " + str(class_names))
    print("test_classes_names --> " + str(test_class_names))
    print("--------------------------------------------------")

    metrics = ['accuracy'] #['categorical_accuracy']

    label_file = os.path.join(_MODELS_DIR, "labels.json")
    print("Loading labels: " + str(label_file))
    with open(label_file) as fp:
        labels = json.load(fp)
        labels.sort()
        print("Labels:" + str(labels))
    print("Labels loaded.")

    if test_class_names != class_names:
        print("FATAL ERROR: training class names are not equal to the test class names")
        exit(-2)
    if labels != class_names:
        print("FATAL ERROR: training class names are not equal to the label names")
        exit(-3)

    image_size = 256 # will be updated by create_model
    run()
