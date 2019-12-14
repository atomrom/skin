# -*- coding: utf-8 -*-
#
# Copyright © 2019 Attila Ulbert
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
# files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, 
# modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software 
# is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from shutil import copyfile
from shutil import copy2
from shutil import SameFileError

from PIL import Image
from scipy import misc

import math
import os
import random
import sys
import codecs
import argparse
import re
import math

TEST_PERCENTAGE = 15
VALIDATION_PERCENTAGE = 15


def list_pictures(directory, ext='jpg|jpeg'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w-]+\.(?:' + ext + '))', f)]


def get_diag(medrec_dir_path):
    f = open(os.path.join(medrec_dir_path, "d.txt"), "r", encoding='utf-8')

    lines = f.readlines()
    if len(lines) == 0:
        return "", None

    line = lines[0]
    print("  " + line)

    diag = line[1:4]

    thickness = None
    thickness_match = re.search(r'(\d*(?:\,\d+)?[ ]?mm)', line)
    if thickness_match is not None:
        thickness = thickness_match[0]

        thickness = thickness[:-2]
        thickness = thickness.replace(",", ".").strip()
        thickness = float(thickness)

    print(thickness)

    return diag, thickness


def copy_and_compress_file(image_file_path, destination_file_path):
    print(image_file_path + " --> " + destination_file_path)

    try:
        img = Image.open(image_file_path)
        exif = []
        if 'exif' in img.info:
            exif = img.info['exif']
        img = img.resize((int(img.size[0] / 3), int(img.size[1] / 3)), Image.LANCZOS)
        if len(exif) > 0:
            img.save(destination_file_path, exif=exif)
        else:
            img.save(destination_file_path)
    except IOError:
        print("IOError: " + image_file_path)



def create_thickness_dataset(root, kind, diags):
    print(kind.upper() + str(diags))

    dataset_file_path = os.path.join(root, kind + ".csv")

    image_dir_path = os.path.join(root, "imgs")
    if not os.path.exists(image_dir_path):
        os.mkdir(image_dir_path)

    with open(dataset_file_path, 'w', encoding='UTF-8') as dataset_file:
        for diag in diags:
            print(diag)

            thickness = diag[0]
            medrec_images = diag[1]

            for image_file_path in medrec_images:
                image_file_name = image_file_path[image_file_path.rfind(os.sep) + 1:]
                destination_file_path = os.path.join(image_dir_path, image_file_name)

                dataset_file.write(str(thickness) + "\t" + image_file_name + "\n")

                copy_and_compress_file(image_file_path, destination_file_path)


def create_thickness_datasets(target_dir, kind, diags):
    kind_dir_path = os.path.join(target_dir, kind)
    os.mkdir(kind_dir_path)

    training_count = int(math.ceil((len(diags) * (100 - VALIDATION_PERCENTAGE - TEST_PERCENTAGE)) / 100))
    training = diags[:training_count]
    create_thickness_dataset(kind_dir_path, "train", training)

    validation_count = int((len(diags) * VALIDATION_PERCENTAGE) / 100)
    validation = diags[training_count: training_count + validation_count]
    create_thickness_dataset(kind_dir_path, "validation", validation)

    test = diags[training_count + validation_count:]
    create_thickness_dataset(kind_dir_path, "test", test)


def create_datasets(target_dir, kind, diags):
    kind_dir_path = os.path.join(target_dir, kind)
    os.mkdir(kind_dir_path)

    for diag in diags:
        print(diag)
        diag_medrecs = diags[diag]

        print(diag_medrecs)

        training_count = int(math.ceil((len(diag_medrecs) * (100 - VALIDATION_PERCENTAGE - TEST_PERCENTAGE)) / 100))
        training = diag_medrecs[:training_count]
        create_dataset(kind_dir_path, "train", diag, training)

        validation_count = int((len(diag_medrecs) * VALIDATION_PERCENTAGE) / 100)
        validation = diag_medrecs[training_count: training_count + validation_count]
        create_dataset(kind_dir_path, "validation", diag, validation)

        test = diag_medrecs[training_count + validation_count:]
        create_dataset(kind_dir_path, "test", diag, test)


def create_dataset(root, kind, diag, image_file_paths):
    print(kind.upper() + str(image_file_paths))

    kind_dir_path = os.path.join(root, kind)
    if not os.path.exists(kind_dir_path):
        os.mkdir(kind_dir_path)

    diag_dir_path = os.path.join(kind_dir_path, diag)
    if not os.path.exists(diag_dir_path):
        os.mkdir(diag_dir_path)

    for medrec_imag_set in image_file_paths:
        for image_file_path in medrec_imag_set:
            image_file_name = image_file_path[image_file_path.rfind(os.sep) + 1:]
            destination_file_path = os.path.join(diag_dir_path, image_file_name)

            copy_and_compress_file(image_file_path, destination_file_path)


def make_thickness_datasets(source_dir, target_dir):
    macro_diags = []
    micro_diags = []
    for patient_dir in os.listdir(source_dir):
        patient_dir_path = os.path.join(source_dir, patient_dir)

        if not os.path.isdir(patient_dir_path):
            print("Not dir: " + patient_dir_path)
            continue

        print("Patient: " + patient_dir_path)

        macro_images = dict()
        micro_images = dict()
        for medrec_dir in os.listdir(patient_dir_path):
            medrec_dir_path = os.path.join(patient_dir_path, medrec_dir)

            print(" Medrec: " + medrec_dir_path)

            macro_images_of_medrec = []
            micro_images_of_medrec = []
            images = list_pictures(medrec_dir_path)
            for image in images:
                print("  Image: " + image)

                image_name_array = image[image.rfind(os.sep) + 1:].split("-")
                image_kind = image_name_array[0]
                if image_kind == 'skinMacroPic':
                    macro_images_of_medrec.append(image)
                elif image_kind == 'skinMicroPic':
                    micro_images_of_medrec.append(image)
                else:
                    print("Error: Unknown image type: " + image_kind)

            _, thickness = get_diag(medrec_dir_path)
            print("  Thickness: " + str(thickness))

            if thickness is None:
                continue

            if len(micro_images_of_medrec) > 0:
                micro_diags.append([thickness, micro_images_of_medrec])

            if len(macro_images_of_medrec) > 0:
                macro_diags.append([thickness, macro_images_of_medrec])

    print(str(micro_diags))
    print()
    print(str(macro_diags))
    print()

    create_thickness_datasets(target_dir, "micro", micro_diags)
    create_thickness_datasets(target_dir, "macro", macro_diags)


def main():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--source_dir",
        type=str,
        help="Path to the root of the directories.")
    parser.add_argument(
        "--target_dir",
        type=str,
        help="Path to the target dir.")
    parser.add_argument(
        "--thickness",
        action='store_true',
        help="Path to the target dir.")

    flags, unparsed = parser.parse_known_args()

    arg_values = ""
    for arg in vars(flags):
        if arg_values is not "":
            arg_values += ","

        print(arg + "='" + str(getattr(flags, arg)) + "'")

    target_dir = flags.target_dir
    source_dir = flags.source_dir

    if flags.thickness:
        make_thickness_datasets(source_dir, target_dir)
    else:
        macro_diags = dict()
        micro_diags = dict()
        for patient_dir in os.listdir(source_dir):
            patient_dir_path = os.path.join(source_dir, patient_dir)

            if not os.path.isdir(patient_dir_path):
                print("Not dir: " + patient_dir_path)
                continue

            print("Patient: " + patient_dir_path)

            macro_images = dict()
            micro_images = dict()
            for medrec_dir in os.listdir(patient_dir_path):
                medrec_dir_path = os.path.join(patient_dir_path, medrec_dir)

                print(" Medrec: " + medrec_dir_path)

                macro_images_of_medrec = []
                micro_images_of_medrec = []
                images = list_pictures(medrec_dir_path)
                for image in images:
                    print("  Image: " + image)

                    image_name_array = image[image.rfind(os.sep)+1:].split("-")
                    image_kind = image_name_array[0]
                    if image_kind == 'skinMacroPic':
                        macro_images_of_medrec.append(image)
                    elif image_kind == 'skinMicroPic':
                        micro_images_of_medrec.append(image)
                    else:
                        print("Error: Unknown image type: " + image_kind)

                diag, _ = get_diag(medrec_dir_path)
                print("  Diag: " + diag)

                if diag == "":
                    continue

                if len(micro_images_of_medrec) > 0:
                    micro_images[medrec_dir] = micro_images_of_medrec

                    if diag in micro_diags:
                        micro_diags[diag].append(micro_images_of_medrec)
                    else:
                        micro_diags[diag] = [micro_images_of_medrec]

                if len(macro_images_of_medrec) > 0:
                    macro_images[medrec_dir] = macro_images_of_medrec

                    if diag in macro_diags:
                        macro_diags[diag].append(macro_images_of_medrec)
                    else:
                        macro_diags[diag] = [macro_images_of_medrec]


        print(str(micro_diags))
        print()
        print(str(macro_diags))
        print()

        create_datasets(target_dir, "micro", micro_diags)
        create_datasets(target_dir, "macro", macro_diags)

    print("Great success!")


if __name__ == '__main__':
    main()
