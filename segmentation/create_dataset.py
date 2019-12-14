# -*- coding: utf-8 -*-

import os
import sys
import codecs
import argparse
import ntpath

from shutil import copy2, SameFileError


def run():
    for file_name in os.listdir(mask_input_dir):
        file_path = os.path.join(mask_input_dir, file_name)

        image_file_name = file_path[file_path.find("mask_")+5:-4]+".jpg"
        category_name = image_file_name[:image_file_name.find("_")]

        image_file_path = os.path.join(image_input_dir, category_name + os.sep + image_file_name)

        target_file_path = os.path.join(dataset_dir, "image")# + os.sep + image_file_name)

        print(file_path)
        print(image_file_path + "->" + target_file_path)
        try:
            copy2(image_file_path, target_file_path)
        except SameFileError:
            continue
        except FileNotFoundError:
            try:
                copy2(image_file_name[:-4]+".JPG", target_file_path)
            except SameFileError:
                continue


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
        "--dry_run",
        type=str2bool,
        default=False,
        help="Don't create masks.")
    parser.add_argument(
        "--mask_input_dir",
        type=str,
        help="Input dir to the masks.")
    parser.add_argument(
        "--image_input_dir",
        type=str,
        help="Input dir to the images.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Dataset dir.")

    flags, unparsed = parser.parse_known_args()

    dry_run = flags.dry_run
    mask_input_dir = flags.mask_input_dir
    image_input_dir = flags.image_input_dir
    dataset_dir = flags.dataset_dir

    for arg in vars(flags):
        print(arg + "='" + str(getattr(flags, arg)) + "'")

    run()
