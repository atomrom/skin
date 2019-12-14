
from model import *
from data import *
from scipy import misc, ndimage

import argparse
import os

from PIL import Image


SAMPLE_SIZE = 121
BATCH_SIZE = 1

CROP_IMG_WIDTH = 256
CROP_IMG_HEIGHT = 256

LESION_MARGIN = 25

MIN_IMG_SIZE = 100
MAX_ASPECT_RATIO = 4

CROP_THRESHOLD = 0.25


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def get_bounding_box(mask):
    width = mask.shape[1]
    height = mask.shape[0]

    x0, x1, y0, y1 = width, -1, height, -1

    found = False
    for x in range(width):
        for y in range(height):
            if mask[x][y] > CROP_THRESHOLD:
                found = True

                if x0 > x:
                    x0 = x
                if x1 < x:
                    x1 = x
                if y0 > y:
                    y0 = y
                if y1 < y:
                    y1 = y

    if x0 < x1 and y0 < y1:
        return found, y0/width, x0/height, y1/width, x1/height
    else:
        return found, 0, 0, 1, 1


def crop(img, y0, x0, y1, x1):
    width = img.shape[1]
    height = img.shape[0]

    y0 = int(y0 * width)
    y1 = int(y1 * width)

    x0 = int(x0 * height)
    x1 = int(x1 * height)

    x0 = max(0, x0 - LESION_MARGIN)
    y0 = max(0, y0 - LESION_MARGIN)
    x1 = min(width, x1 + LESION_MARGIN)
    y1 = min(height, y1 + LESION_MARGIN)

    too_small = (x1 - x0) < MIN_IMG_SIZE or (y1 - y0) < MIN_IMG_SIZE

    too_elongated = True
    if x0 != x1 and y0 != y1:
        aspect_ratio = (x1 - x0) / (y1 - y0)
        too_elongated = aspect_ratio >= MAX_ASPECT_RATIO or 1/aspect_ratio >= MAX_ASPECT_RATIO

    print("x0:" + str(x0) + ", y0:" + str(y0) + ", x1:" + str(x1) + ", y1:" + str(y1))

    return too_small, too_elongated, img[x0:x1, y0:y1, :]


def generate_segmentation_mask(img):
    img = rgb2gray(img)
    img = trans.resize(img, (CROP_IMG_WIDTH, CROP_IMG_HEIGHT))
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img, (1,) + img.shape)

    mask = model.predict(img, verbose=1)[0]
    mask = mask[:, :, 0]

    return mask


def generate_cropped_image():
    for category in os.listdir(test_dir):
        category_dir_path = os.path.join(test_dir, category)
        if not os.path.isdir(category_dir_path):
            continue

        for img_file in os.listdir(category_dir_path):
            img_file_path = os.path.join(category_dir_path, img_file)

            out_category_dir = os.path.join(test_results_dir, category)
            if not os.path.exists(out_category_dir):
                os.mkdir(out_category_dir)

            out_file_path = os.path.join(out_category_dir, "crop_" + img_file)
            mask_file_path = os.path.join(out_category_dir, "mask_" + img_file[:-4] + ".png")

            loaded_img = np.array(Image.open(img_file_path))

            mask = generate_segmentation_mask(loaded_img)
            found, y0, x0, y1, x1 = get_bounding_box(mask)

            if found:
                too_small, too_elongated, cropped_img = crop(loaded_img, y0, x0, y1, x1)
                if not too_small and not too_elongated:
                    misc.imsave(out_file_path, cropped_img)
                elif too_small:
                    print("Too small -> " + img_file)
                elif too_elongated:
                    print("Too elongated -> " + img_file)
            else:
                print("Not found! -> " + img_file)

            if save_mask:
                misc.imsave(mask_file_path, mask)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--save_mask",
        type=str2bool,
        default=False,
        help="Save segmentation mask.")
    parser.add_argument(
        "--model_weights_file",
        type=str,
        default="checkpoints/model.h5",
        help="Model weights.")
    parser.add_argument(
        "--test_dir",
        type=str,
        default="dataset/test",
        help="Root directory of test files.")
    parser.add_argument(
        "--test_results_dir",
        type=str,
        default="data/test_results",
        help="Root directory of test results.")

    flags, unparsed = parser.parse_known_args()

    test_dir = flags.test_dir
    test_results_dir = flags.test_results_dir
    model_weights_file = flags.model_weights_file
    save_mask = flags.save_mask

    print("test images : " + test_dir)
    print("test results: " + test_results_dir)

    model = unet()
    model.load_weights(model_weights_file)

    #testGene = testGenerator(test_dir)
    #results = model.predict_generator(testGene, int(SAMPLE_SIZE/BATCH_SIZE), verbose=1)
    #saveResult(test_results_dir, results)

    generate_cropped_image()