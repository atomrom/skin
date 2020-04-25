import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import sys
import time
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from color_transfer import color_transfer

import align

from keras.preprocessing import image


from keras.models import load_model, Model

SUBIMAGE_SIZE = 1024

LESION_BOX_MARGIN = 5
LESION_MARGIN = 20
MIN_LESION_IMAG_SIZE = 128

changed_html_text = ""
not_changed_html_text = ""

class InferenceConfig(Config):
    # Give the configuration a recognizable name
    NAME = "shapes"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 lesion

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    #IMAGE_RESIZE_MODE = "none"
    IMAGE_RESIZE_MODE = "square"

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

inference_config = InferenceConfig()


def get_subimage(img_data, x0, y0, x1, y1):
    print(y0,y1,x0,x1)
    print(str(img_data.shape))

    y0 = int(np.max([0, y0]))
    x0 = int(np.max([0, x0]))
    # y1 = int(np.min([img_data.shape[0], y1]))
    # x1 = int(np.min([img_data.shape[1], x1]))

    return img_data[y0:y1, x0:x1, :]


def get_all_image_lesion_bboxes(image_lesions):
    image_lesion_rois = []

    for subimage_lesions in image_lesions:
        x = subimage_lesions["x"]
        y = subimage_lesions["y"]

        lesions = subimage_lesions["lesions"]
        rois = lesions['rois']
        for roi in rois:
            x0 = x + roi[1]
            y0 = y + roi[0]
            x1 = x + roi[3]
            y1 = y + roi[2]

            image_lesion_rois.append([x0, y0, x1, y1])

    return image_lesion_rois



def find_lesions(img_data, prefix):
    # img = image.load_img("c:/temp/70.jpg ")
    # img = image.load_img("c:/Users/eattulb/Downloads/change/img_020.JPG")#, target_size=(image_size, image_size))

    img_data = img_data.copy()

    (image_height, image_width, _) = img_data.shape

    index = 0

    lesion_masks = []
    bboxes = []
    for x in range(0, image_width, SUBIMAGE_SIZE):
        for y in range(0, image_height, SUBIMAGE_SIZE):
            # x = 1024
            # y = 1024
            print(x, y)

            subimage = get_subimage(img_data, x, y, x+SUBIMAGE_SIZE, y+SUBIMAGE_SIZE)
            results = model.detect([subimage], verbose=1)
            r = results[0]

            lesion_masks.append({"x": x, "y": y, "lesions": r})

            rois = r['rois']
            for roi in rois:
                print(str(roi))

                index += 1

                y0, x0, y1, x1 = roi
                bboxes.append([x + roi[1], y + roi[0], x + roi[3], y + roi[2]])

                color = (0, 200, 0)
                # cv2.putText(subimage, str(index), (x0-2*LESION_BOX_MARGIN, y0-2*LESION_BOX_MARGIN), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)
                # subimage = visualize.draw_box(subimage, (y0-LESION_BOX_MARGIN, x0-LESION_BOX_MARGIN, y1+LESION_BOX_MARGIN, x1+LESION_BOX_MARGIN), color, thickness=3)

                cv2.putText(img_data, str(index), (x + x0-2*LESION_BOX_MARGIN, y + y0-2*LESION_BOX_MARGIN), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)
                img_data = visualize.draw_box(img_data, (y + y0-LESION_BOX_MARGIN, x + x0-LESION_BOX_MARGIN, y + y1+LESION_BOX_MARGIN, x + x1+LESION_BOX_MARGIN), color, thickness=3)

    img_data = img_data.astype(np.uint8)
    if prefix is not None:
        image_file_path = os.path.join(target_dir, prefix + "_img.jpg")
        Image.fromarray(img_data).save(image_file_path)

    return lesion_masks, bboxes, img_data


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    print("MED:" + str(np.average(lab_planes[0])))

    return img


def lightness(img, average):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if average is not None:
        lightness_plane = lab_planes[0] #np.array(lab_planes[0], dtype='float')
        # lightness_plane = clahe.apply(lightness_plane)
        # lab_planes[0] = lightness_plane

        diff = average - np.median(lightness_plane)
        if diff > 0:
            diff = np.uint8(diff)

            lim = 255 - diff
            lightness_plane[lightness_plane > lim] = 255
            lightness_plane[lightness_plane <= lim] += diff
        else:
            diff = np.uint8(-diff)

            lightness_plane[lightness_plane < diff] = 0
            lightness_plane[lightness_plane >= diff] -= diff
    # else:
    #     lightness_plane = lab_planes[0]  # np.array(lab_planes[0], dtype='float')
    #     lightness_plane = clahe.apply(lightness_plane)
    #     lab_planes[0] = lightness_plane

    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img, int(np.average(lab_planes[0]))


def load_images():
    # Read reference image
    print("Reading reference image: ", old_image_path)
    imReference = cv2.imread(old_image_path, cv2.IMREAD_COLOR)

    imReference, average = lightness(imReference, None)

    old_file_name = old_image_path[old_image_path.rfind(os.sep) + 1:]
    old_target_file_path = os.path.join(target_dir, old_file_name)
    cv2.imwrite(old_target_file_path, imReference)

    print(old_file_name)
    print(old_target_file_path)

    # Read image to be aligned
    print("Reading image to align: ", new_image_path);
    im = cv2.imread(new_image_path, cv2.IMREAD_COLOR)

    im, _ = lightness(im, average)
    im = color_transfer(imReference, im)

    new_file_name = new_image_path[new_image_path.rfind(os.sep) + 1:]
    new_target_file_path = os.path.join(target_dir, new_file_name)
    cv2.imwrite(new_target_file_path, im)

    # return cv2.cvtColor(imReference, cv2.COLOR_BGR2RGB), cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return imReference, im, old_file_name, new_file_name


def align_images():
    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    aligned_new_image, h = align.alignImages(new_image, old_image)

    # # Write aligned image to disk.
    # aligned_new_file_path = os.path.join(target_dir, "aligned_new.jpg")
    #
    # print("Saving aligned new image: ", aligned_new_file_path);
    # cv2.imwrite(aligned_new_file_path, aligned_new_image)

    # Print estimated homography
    print("Estimated homography: \n", h)

    diff_image = cv2.absdiff(old_image, aligned_new_image)

    diff_file_path = os.path.join(target_dir, "diff.jpg")
    cv2.imwrite(diff_file_path, diff_image)

    # return cv2.cvtColor(aligned_new_image, cv2.COLOR_BGR2RGB), cv2.cvtColor(diff_image, cv2.COLOR_BGR2RGB)

    return aligned_new_image, diff_image


def crop_and_save_lesion(image_data, filename, x, y, roi):
    sx = roi[3] - roi[1] + 2 * LESION_MARGIN
    sy = roi[2] - roi[0] + 2 * LESION_MARGIN

    s = int(np.max([sx, sy, MIN_LESION_IMAG_SIZE]) / 2)

    cx = int(x + (roi[1] + roi[3]) / 2)
    cy = int(y + (roi[0] + roi[2]) / 2)

    lesion_data = get_subimage(image_data, cx - s, cy - s, cx + s, cy + s)

    lesion_file_path = os.path.join(target_dir, filename)
    Image.fromarray(lesion_data.astype(np.uint8)).save(lesion_file_path)

    return lesion_data


def crop_and_save_lesions(old_image_data, new_image_data, image_lesions, nearest_lesion_indices):
    global changed_html_text
    global not_changed_html_text

    index = 0

    for subimage_lesions in image_lesions:
        x = subimage_lesions["x"]
        y = subimage_lesions["y"]
        lesions = subimage_lesions["lesions"]

        rois = lesions['rois']
        for roi in rois:
            index = index + 1
            print("index: " + str (index))
            print(str(roi))

            diff = ""
            color = "black"
            if index in nearest_lesion_indices:
                diff = "DIFF_"
                color = "red"

            old_filename = "old_" + diff + str(index) + ".jpg"
            new_filename = "new_" + diff + str(index) + ".jpg"
            old_lesion_data = crop_and_save_lesion(old_image_data, old_filename, x, y, roi)
            new_lesion_data = crop_and_save_lesion(new_image_data, new_filename, x, y, roi)

            old_features = predict(old_lesion_data)
            new_features = predict(new_lesion_data)

            distance = np.linalg.norm(old_features - new_features)

            print(str(old_features))
            print(str(new_features))
            print("distance="+str(distance))

            # dist = np.linalg.norm(old_lesion_data - new_lesion_data)
            # print("dist: " + str(dist))

            html_text = "<p><font color=\"" + color + "\">" + str(index) + ".</font></p>\n" \
                        "<img src=\"" + old_filename + "\" alt=\"diff\" width=\"128\" >\n" \
                        "<img src=\"" + new_filename + "\" alt=\"diff\" width=\"128\" >\n" \
                        "d=" + str(distance) + "\n"
            if diff is "":
                not_changed_html_text += html_text
            else:
                changed_html_text += html_text


def get_all_image_lesion_bboxes(image_lesions):
    image_lesion_rois = []

    for subimage_lesions in image_lesions:
        x = subimage_lesions["x"]
        y = subimage_lesions["y"]

        lesions = subimage_lesions["lesions"]
        rois = lesions['rois']
        for roi in rois:
            x0 = x + roi[1]
            y0 = y + roi[0]
            x1 = x + roi[3]
            y1 = y + roi[2]

            image_lesion_rois.append([x0, y0, x1, y1])

    return image_lesion_rois


def find_nearest_bbox_index(bbox, all_image_lesion_bboxes):
    nearest_index = None
    smallest_distance = np.inf

    for i in range(len(all_image_lesion_bboxes)):
        d = np.linalg.norm(np.array(bbox) - np.array(all_image_lesion_bboxes[i]))
        if d < smallest_distance:
            smallest_distance = d
            nearest_index = i + 1

    return nearest_index


def find_nearest_lesions(diff_lesions, image_lesions):
    lesion_indices = set()

    all_image_lesion_bboxes = get_all_image_lesion_bboxes(image_lesions)

    for diff_subimage in diff_lesions:
        x = diff_subimage["x"]
        y = diff_subimage["y"]

        lesions = diff_subimage["lesions"]
        rois = lesions['rois']
        for roi in rois:
            x0 = x + roi[1]
            y0 = y + roi[0]
            x1 = x + roi[3]
            y1 = y + roi[2]

            nearest_bbox_index = find_nearest_bbox_index([x0, y0, x1, y1], all_image_lesion_bboxes)
            if nearest_bbox_index is not None:
                lesion_indices.add(nearest_bbox_index)

    return lesion_indices


def highlight_with_diff_and_save_image(image_data, bboxes, nearest_lesion_indices, prefix):
    for index in nearest_lesion_indices:
        color = (200, 0, 0) # red

        x0, y0, x1, y1 = bboxes[index - 1]

        cv2.putText(image_data, str(index),
                    (x0-2*LESION_BOX_MARGIN, y0-2*LESION_BOX_MARGIN),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)
        visualize.draw_box(image_data,
                           (y0-LESION_BOX_MARGIN, x0-LESION_BOX_MARGIN, y1+LESION_BOX_MARGIN, x1+LESION_BOX_MARGIN),
                           color, thickness=3)

    image_file_path = os.path.join(target_dir, prefix + "_img.jpg")
    Image.fromarray(image_data).save(image_file_path)


def predict(input_image):
    img = cv2.resize(input_image, (221, 221))
    #img = image.load_img(input_file, target_size=(image_size, image_size))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    # img_data = preprocess_input(img_data)
    feature = cnn_model.predict(img_data)

    return feature


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--old_image",
        type=str,
        help="Path to the old image.")
    parser.add_argument(
        "--new_image",
        type=str,
        help="Path to the old image.")
    parser.add_argument(
        "--smodel",
        type=str,
        help="Path to the rcnn model.")
    parser.add_argument(
        "--cmodel",
        type=str,
        help="Path to the convolutional model.")
    parser.add_argument(
        "--target_root",
        type=str,
        help="Report dir root.")

    flags, unparsed = parser.parse_known_args()

    old_image_path = flags.old_image
    new_image_path = img_width = image_size = flags.new_image
    segmentation_model_path = flags.smodel
    convolutional_model_path = flags.cmodel
    target_root = flags.target_root

    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S", time.gmtime())
    target_dir = os.path.join(target_root, timestamp)

    if not os.path.exists(target_dir):
        print("Making target dir: " + target_dir)
        os.mkdir(target_dir)

    for arg in vars(flags):
        print(arg + "='" + str(getattr(flags, arg)) + "'")

    print("Loading weights from ", convolutional_model_path)
    cnn_model = load_model(convolutional_model_path)
    print("Model loaded.")

    image_size = int(cnn_model.input.shape[1])
    inp = cnn_model.input
    out = cnn_model.layers[-6].output
    cnn_model = Model(inp, out)

    # reference_image_feature = predict(image_path, image_size, model)

    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=segmentation_model_path)

    print("Loading weights from ", segmentation_model_path)
    model.load_weights(segmentation_model_path, by_name=True)
    print("Model loaded.")

    html_file_path = os.path.join(target_dir, "index.html")
    html_file = open(html_file_path, "w", encoding='utf-8')

    html_file.write("<html><head><title>Lesion report</title></head><body>\n")

    old_image, new_image, old_image_file_path, new_image_file_path = load_images()

    aligned_new_image, diff_image = align_images()

    old_image = cv2.cvtColor(old_image, cv2.COLOR_BGR2RGB)
    aligned_new_image = cv2.cvtColor(aligned_new_image, cv2.COLOR_BGR2RGB)
    aligned_new_image_data = image.img_to_array(aligned_new_image)

    new_image_lesions, new_image_lesion_bboxes, annotated_new_image_data = find_lesions(aligned_new_image_data, None)
    diff_image_lesions, _, _ = find_lesions(image.img_to_array(diff_image), "diff")

    old_image_data = image.img_to_array(old_image)

    nearest_lesion_indices = find_nearest_lesions(diff_image_lesions, new_image_lesions)
    print("found:" + str(nearest_lesion_indices))

    highlight_with_diff_and_save_image(annotated_new_image_data, new_image_lesion_bboxes, nearest_lesion_indices, "new")

    crop_and_save_lesions(old_image_data, aligned_new_image_data, new_image_lesions, nearest_lesion_indices)
    # crop_and_save_lesions(old_image_data, aligned_new_image_data, diff_image_lesions)

    html_file.write("<h1>Old and new</h1>")
    html_file.write("<img src=\"" + old_image_file_path + "\" alt=\"old\" width=\"500\" >\n")
    html_file.write("<img src=\"" + new_image_file_path + "\" alt=\"old\" width=\"500\" >\n")

    html_file.write("<h1>Diff</h1>\n")
    html_file.write("<img src=\"diff.jpg\" alt=\"diff\" width=\"700\" >\n")

    html_file.write("<h1>Lesion map</h1>\n")
    html_file.write("<img src=\"new_img.jpg\" alt=\"diff\" width=\"700\" >\n")

    html_file.write("<h1>Lesions</h1>\n")
    html_file.write("<h2>Changed (old vs new)</h2>\n")

    html_file.write(changed_html_text)

    html_file.write("<h2>Not changed</h2>\n")

    html_file.write(not_changed_html_text)

    html_file.write("</body></html>\n")