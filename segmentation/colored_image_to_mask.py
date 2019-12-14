# -*- coding: utf-8 -*-

import os
import sys
import codecs
import argparse
import ntpath
import cv2

import numpy as np

from PIL import Image, ImageFilter


def get_filename(path):
    head, tail = ntpath.split(path)

    return tail or ntpath.basename(head)


def create_mask_file(file_path):
    file_name = get_filename(file_path)

    if file_name.startswith("mask"):
        return

    dest_file_name = file_name[:-4]
    while not dest_file_name[-1:].isdigit():
        #print(dest_file_name)
        dest_file_name = dest_file_name[:-1]

    destination_file = os.path.join(output_dir, "mask_" + dest_file_name + ".png")

    print(file_path + " -> " + destination_file)

    if dry_run:
        return

    img = Image.open(file_path).convert('L')

    img = img.point(lambda x: 0 if x < 255 else 255, '1')
    img.filter(ImageFilter.MaxFilter(3))

    img.save(destination_file)


def run():
    if os.path.isdir(input_path):
        for file_name in os.listdir(input_path):
            file_path = os.path.join(input_path, file_name)

            create_mask_file(file_path)
    else:
        create_mask_file(input_path)


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
        "--input_path",
        type=str,
        help="Path to a colored image or directory of images.")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Root directory output file(s).")

    flags, unparsed = parser.parse_known_args()

    dry_run = flags.dry_run
    input_path = flags.input_path
    output_dir = flags.output_dir

    print("dry_run: '" + str(dry_run) + "'")
    print("input_path: '" + input_path + "'")
    print("output_dir: '" + output_dir + "'")

    run()
