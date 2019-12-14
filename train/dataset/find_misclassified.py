import itertools
import glob
import os
import numpy as np
import keras
import pickle
import csv
import sys
import codecs
import argparse

import matplotlib.pyplot as plt

import predict as pred

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report

from scipy import interp
from itertools import cycle

from PIL import Image

import keras.preprocessing.image as image
from keras.applications.nasnet import NASNetMobile, NASNetLarge, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import backend as K


def load_models(model_dir):
    models = []

    for file in os.listdir(model_dir):
        if file.endswith(".model"):
            file = os.path.join(model_dir, file)

            print("Loading model and weights: " + file)
            model = load_model(file)
            models.append(model)
            print("Model loaded.")

    return models


def _model_predict(model, pil_img):
    image_size = int(model.input.shape[1])

    img = pil_img.resize((image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    prediction_array = model.predict(x)[0]

    return prediction_array


def get_prediction(models, filepath):
    print("Predicting: " + filepath)

    pil_img = Image.open(filepath)

    prediction_array = []
    for model in models:
        if len(prediction_array) == 0:
            prediction_array = _model_predict(model, pil_img)
        else:
            prediction_array += _model_predict(model, pil_img)

    pred = np.argmax(prediction_array)
    print(str(pred))

    return pred


def array_to_csv(array):
    csv = ""

    for item in array:
        if csv is not "":
            csv += ","

        csv += str(item)

    return csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model file or directory containing several models files.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory to the dataset [train|test|validation].")
    parser.add_argument(
        "--preds_path",
        type=str,
        help="Path to the misclassified image log.")

    flags, unparsed = parser.parse_known_args()

    model_path = flags.model_path
    dataset_dir = flags.dataset_dir
    preds_path = flags.preds_path

    for arg in vars(flags):
        print(arg + "='" + str(getattr(flags, arg)) + "'")

    models = []
    if os.path.isdir(model_path):
        models = load_models(model_path)
    else:
        models.append(load_model(model_path))

    preds_file = None
    if preds_path is not None:
        preds_file = open(preds_path, "w")

    dataset_dir = os.path.abspath(dataset_dir)

    category_index = 0
    for category_dir in sorted(os.listdir(dataset_dir)):
        category_path = os.path.join(dataset_dir, category_dir)

        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)

            prediction_class_index = get_prediction(models, image_path)

            if category_index != prediction_class_index:
                print("Misclassified!!! " + str(category_index) + " != " + str(prediction_class_index))
                entry = image_path + "\n"
                if preds_file is not None:
                    preds_file.write(entry)
                else:
                    print(entry)

        category_index += 1

    print("Great success!")
