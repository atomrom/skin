from model import *
from data import *

import argparse

from keras.callbacks import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/model.h5",
        help="Path to the model.")
    parser.add_argument(
        "--test_dir",
        type=str,
        help="Root directory of test files.")

    flags, unparsed = parser.parse_known_args()

    model_path = flags.model_path
    test_dir = flags.test_dir

    print("model path :" + flags.model_path)
    print("test       :" + str(flags.test_dir))

    model = unet(pretrained_weights=model_path)

    data_gen_args = dict(rotation_range=0,
                         horizontal_flip=False,
                         vertical_flip=False,
                         fill_mode='nearest')

    testGene = trainGenerator(1, test_dir, 'images', 'masks', data_gen_args)
    results = model.evaluate_generator(testGene, steps=80)

    print(str(model.metrics_names) + "=" + str(results))

    # saveResult(test_dir + '/masks/VASC', results)

