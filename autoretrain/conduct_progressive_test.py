# -*- coding: utf-8 -*-
import os
import numpy as np
import argparse

from PIL import Image

import keras.preprocessing.image as image
from keras.models import load_model

import create_clinic_datasets as ccd


num_classes = None

# isic18
# class_names = [AKIEC  BCC  ,"BKL","DF","MEL","NV","VASC"]

# munich
# class_names = ["m", "n"]

class_name_dict = {
    "macro": ["C43", "C44", "C79", "D03", "D22", "D23", "L30", "L82", "L90"],
    "micro": ["C43", "C44", "D03", "D22", "D23", "L30", "L82", "L90"]
}

# micro-bno10
# class_names = ["C43", "C44", "D03", "D22", "D23", "L30", "L82", "L90"]

not_diagnosed_count = 0

misclassified = []
unconfident = []

img_height, img_width = 450, 450  # 224, 224

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


def get_prediction_array(models, filepath, number_of_classes):
    print(filepath)

    loaded_img = Image.open(filepath)

    prediction_array = [0] * number_of_classes
    for model_idx in range(len(models)):
        img = loaded_img

        img = img.resize((256, 256))
        img = image.img_to_array(img).astype('float32')
        #img /= 255.
        img = np.expand_dims(img, axis=0)

        model = models[model_idx]
        pa = model.predict(img)[0]

        for i in range(number_of_classes):
            prediction_array[i] += pa[i]

    return prediction_array


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
        "--mel_bias",
        type=float,
        help="Diagnose melanoma if its weight reaches this value.")
    parser.add_argument(
        "--mel_idx",
        type=int,
        help="Index of melanoma in the prediction array.")
    parser.add_argument(
        "--patients_dir",
        type=str,
        help="Directory to the patient image dirs to diagnose.")
    parser.add_argument(
        "--preds_prefix_path",
        type=str,
        help="Prefix path to the predictions CSVs.")
    parser.add_argument(
        "--kind",
        type=str,
        help="micro | macro")
    parser.add_argument(
        "--diags_path",
        type=str,
        help="Path to the diags.csv file, output of parse_diag.py")

    flags, unparsed = parser.parse_known_args()

    model_path = flags.model_path
    mel_bias = flags.mel_bias
    mel_idx = flags.mel_idx
    patients_dir = flags.patients_dir
    preds_prefix_path = flags.preds_prefix_path
    kind = flags.kind
    diags_path = flags.diags_path

    class_names = class_name_dict[kind]

    for arg in vars(flags):
        print(arg + "='" + str(getattr(flags, arg)) + "'")

    models = []
    if os.path.isdir(model_path):
        models = load_models(model_path)
    else:
        models.append(load_model(model_path))

    f = None
    patient_csv = None
    if preds_prefix_path is not None:
        f = open(preds_prefix_path + "_" + kind + "_images.csv", "w", encoding='utf-8')
        patient_csv = open(preds_prefix_path + "_" + kind + "_patients.csv", "w", encoding='utf-8')

    number_of_classes = models[0].output_shape[1]

    diags = ccd.load_diags(diags_path)

    i = 1
    for image_dir in os.listdir(patients_dir):
        patient_dir_path = os.path.join(patients_dir, image_dir)
        if not os.path.isdir(patient_dir_path):
            continue

        cumulative_prediction_array = [0] * number_of_classes

        patient_name = image_dir[:image_dir.index(",")]

        sample_date = ccd.extract_sample_date(image_dir)

        for file in os.listdir(patient_dir_path):
            if kind not in file:
                continue

            image_path = os.path.join(patient_dir_path, file)

            prediction_array = get_prediction_array(models, image_path, len(class_names))

            cumulative_prediction_array = np.add(cumulative_prediction_array, prediction_array)

            pred = 0
            if mel_idx is not None:
                if prediction_array[mel_idx]/num_classes > mel_bias:
                    pred = mel_idx
                else:
                    pred = np.argmax(prediction_array)
            else:
                pred = np.argmax(prediction_array)

            print(str(i) + ". " + file + "-->" + class_names[pred] + " - " + str(prediction_array))

            if f is not None:
                f.write(str(i) + "," + file + "," + class_names[pred] + "," + array_to_csv(prediction_array) + "\n")

            i += 1

        cumulative_prediction_class = None
        cumulative_prediction = np.argmax(cumulative_prediction_array)
        if cumulative_prediction_array[cumulative_prediction] != 0:
            cumulative_prediction_class = class_names[cumulative_prediction]

        print(str(i))

        histology, supposed = ccd.get_diag(diags, image_dir)

        if patient_csv is not None:
            patient_csv.write(patient_name + "," + str(sample_date) + "," +
                              str(histology) + "," + str(supposed) + "," +
                              str(cumulative_prediction_class) + "," +
                              array_to_csv(cumulative_prediction_array) +
                              "\n")
