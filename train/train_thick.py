# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten, GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model 

from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3

from sklearn.model_selection import train_test_split
import numpy as np

import locale
import os
import argparse
import csv
import cv2
import keras

MAX_THICKNESS = 9.0
MAX_EPOCHS = 200

num_frozen_layers = 0
num_fc_neurons = 4096
dropout = 0.5
patience = 20

def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    chanDim = -1

    base_model = InceptionV3(#DenseNet121(
            input_shape=(height, width, depth),
            include_top=False,
            weights='imagenet')

    x = base_model.output
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = GlobalAveragePooling2D()(x)
    #x = Dense(num_fc_neurons, activation='relu')(x)
    #x = Dropout(dropout)(x)
    #x = Dense(num_fc_neurons, activation='relu')(x)
    #x = Dropout(dropout)(x)
    #predictions = Dense(nb_classes, activation='softmax')(x)
    #model = Model(inputs=base_model.input, outputs=predictions)

     # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(base_model.input, x)


    i = 0
    for layer in model.layers: #conv_model.layers:
        i += 1
        if i <= num_frozen_layers:
            layer.trainable = False
        else:
            layer.trainable = True

        print('{0}:\t{1}'.format(layer.trainable, layer.name))

    return model

    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1

    # define the model input
    inputs = Input(shape=inputShape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model


def load_dataset(kind, dataset_dir):
    print("Loading thicknesses...")
    thickness_file = os.path.join(dataset_dir, kind + ".csv")
    print(thickness_file)

    print("Loading images...")
    thicknesses = []
    images = []
    with open(thickness_file) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')

        for row in reader:
            image_name = row[1]
            thickness = row[0]

            print(image_name + " --> " + thickness)
    
            thickness = float(thickness)        
            if thickness > MAX_THICKNESS:
                print("Too thick!!! " + str(thickness))
                continue

            image_path = os.path.join(dataset_dir, "imgs", image_name)
            if not os.path.exists(image_path):
                print("Missing!")
                continue

            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_size, image_size))

            thicknesses.append(thickness / MAX_THICKNESS)
            images.append(image / 255.0)

    #thicknesses /= MAX_THICKNESS
    #images /= 255.0

    return np.asarray(thicknesses), np.asarray(images)


def train():
    save_weights_path = os.path.join("models", experiment_id + ".model")
    if os.path.exists(save_weights_path):
        print("Already exists: " + save_weights_path)
        return

    train_thicknesses, train_images = load_dataset("train", dataset_dir)
    validation_thicknesses, validation_images = load_dataset("validation", dataset_dir)
    test_thicknesses, test_images = load_dataset("test", dataset_dir)

    model = create_cnn(image_size, image_size, 3, regress=True)
    opt = Adam(lr=learning_rate)#, decay=1e-3 / MAX_EPOCHS)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    checkpointer = keras.callbacks.ModelCheckpoint(filepath=save_weights_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')

    # train the model
    print("Training model...")
    model.fit(train_images, train_thicknesses, validation_data=(validation_images, validation_thicknesses),
              epochs=epochs, batch_size=8, callbacks=[checkpointer, earlyStopping],)
    # make predictions on the testing data
    print("Testing...")
    model = load_model(save_weights_path)
    preds = model.predict(test_images)    

    hit = [1] * len(test_thicknesses)

    # compute the difference between the *predicted* house prices and the
    # *actual* house prices, then compute the percentage difference and
    # the absolute percentage difference
    diff = preds.flatten() - test_thicknesses
    percentDiff = (diff / test_thicknesses) * 100
    absPercentDiff = np.abs(percentDiff)

    for i in range(len(diff)):
        p = preds[i][0] * MAX_THICKNESS
        t = test_thicknesses[i] * MAX_THICKNESS
        if (p >= 1 and t < 1) or (p < 1 and t >= 1):
            hit[i] = 0

        print(str(p) + "\t" + str(t) + "\t" + str(absPercentDiff[i]) + "\t" + str(hit[i]))

    # compute the mean and standard deviation of the absolute percentage
    # difference
    mean = np.mean(absPercentDiff)
    std = np.std(absPercentDiff)
    hitPercent = (np.sum(hit) / len(hit)) * 100

    # finally, show some statistics on our model
    #print("avg. thickness: {}, std thickness: {}".format(
    #    df["price"].mean(),
    #    df["price"].std()))
    print("mean: {:.2f}%, std: {:.2f}%, hit: {:.2f}%".format(mean, std, hitPercent))

    #print("Saving model...")
    #model.save(save_weights_path)
    #print("Model saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Path to the dataset dir.")
    parser.add_argument(
        "--image_size",
        type=int,
        help="Image size.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=MAX_EPOCHS,
        help="Number of epochs")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate")
    parser.add_argument(
        "id",
        type=str,
        help="Experiment id.")

    flags, unparsed = parser.parse_known_args()

    arg_values = ""
    for arg in vars(flags):
        if arg_values is not "":
            arg_values += ","

        print(arg + "='" + str(getattr(flags, arg)) + "'")

    dataset_dir = flags.dataset_dir
    image_size = flags.image_size
    experiment_id = flags.id
    epochs = flags.epochs
    learning_rate = flags.learning_rate

    train()

