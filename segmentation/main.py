from model import *
from data import *

import argparse

from keras.callbacks import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CHECKPOINTS_DIR = 'checkpoints/'

EPOCHS = 9999999999999999

SAMPLE_SIZE = 191
VALIDATION_SAMPLE_SIZE = 51

BATCH_SIZE = 1

PATIENCE = 10

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda v: v.lower() == "true")
	parser.add_argument(
		"--training_dir",
		type=str,
		default="dataset/training",
		help="Root directory of training files.")
	parser.add_argument(
		"--validation_dir",
		type=str,
		default="dataset/validation",
		help="Root directory of validation files.")
	# parser.add_argument(
	# 	"--test_dir",
	# 	type=str,
	# 	help="Root directory of test files.")

	flags, unparsed = parser.parse_known_args()

	training_dir = flags.training_dir
	validation_dir = flags.validation_dir
	# test_dir = flags.test_dir

	print("training    :" + flags.training_dir)
	print("validation  :" + flags.validation_dir)
	# print("test:" + str(flags.test_dir))

	data_gen_args = dict(rotation_range=1,
						horizontal_flip=True,
						vertical_flip=True,
						fill_mode='nearest')
	test_gen_args = dict()
	trainGen = trainGenerator(BATCH_SIZE, training_dir, 'images', 'masks', data_gen_args, save_to_dir=None)
	validationGen = trainGenerator(BATCH_SIZE, validation_dir, 'images', 'masks', data_gen_args, save_to_dir=None)

	modelCheckpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True)
	earlyStopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, mode='auto')

	model = unet(pretrained_weights='old_model.h5')
	model.fit_generator(trainGen, steps_per_epoch=int(SAMPLE_SIZE / BATCH_SIZE), epochs=EPOCHS, verbose=1, callbacks=[earlyStopping, modelCheckpoint], validation_data = validationGen, validation_steps=int(VALIDATION_SAMPLE_SIZE / BATCH_SIZE))
	
	# testGene = testGenerator(test_dir+'/images/AKIEC')
	# results = model.predict_generator(testGene,327,verbose=1)
	# saveResult(test_dir+'/masks/AKIEC', results)
	# testGene = testGenerator(test_dir+'/images/BCC')
	# results = model.predict_generator(testGene,514,verbose=1)
	# saveResult(test_dir+'/masks/BCC', results)
	# testGene = testGenerator(test_dir+'/images/BKL')
	# results = model.predict_generator(testGene,1099,verbose=1)
	# saveResult(test_dir+'/masks/BKL', results)
	# testGene = testGenerator(test_dir+'/images/DF')
	# results = model.predict_generator(testGene,115,verbose=1)
	# saveResult(test_dir+'/masks/DF', results)
	# testGene = testGenerator(test_dir+'/images/MEL')
	# results = model.predict_generator(testGene,1113,verbose=1)
	# saveResult(test_dir+'/masks/MEL', results)
	# testGene = testGenerator(test_dir+'/images/NV')
	# results = model.predict_generator(testGene,6704,verbose=1)
	# saveResult(test_dir+'/masks/NV', results)
	# testGene = testGenerator(test_dir+'/images/VASC')
	# results = model.predict_generator(testGene,142,verbose=1)
	# saveResult(test_dir+'/masks/VASC', results)

