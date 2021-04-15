# import the necessary packages
# set the matplotlib backend so figures can be saved in the background

import matplotlib
matplotlib.use("Agg")

import sys
sys.path.append('../')

# import necessary Packages
import json
import os
from dogs_vs_cats.config import dogs_vs_cats_config
from Models.alexnet import AlexNet
from IO.hdf5datasetgenerator import HDF5DatasetGenerator
from Preprocessing.Preprocessor import Preprocess
from Callbacks.trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# load the RGB means for the training set
means = json.loads(open(dogs_vs_cats_config.DATASET_MEAN).read())

# initialize the image preprocessors
p = Preprocess(height=227, width=227, rMean=means["R"], gMean=means["G"], bMean=means["B"])
sp = p.resize
pp = p.Patch_preprocessor
iap = p.image_to_array
mp = p.Mean_preprocessor

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(dogs_vs_cats_config.TRAIN_HDF5, 128, aug=aug, preprocessors=[pp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(dogs_vs_cats_config.VAL_HDF5, 128, preprocessors=[sp, mp, iap], classes=2)

# initialize the optimizer
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# construct the set of callbacks
path = os.path.sep.join([dogs_vs_cats_config.OUTPUT_PATH, "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]

# train the network
model.fit_generator(trainGen.generator(), steps_per_epoch=trainGen.numImages // 128,
                    validation_data=valGen.generator(), validation_steps=valGen.numImages // 128,
                    epochs=75, max_queue_size=128 * 2, callbacks=callbacks, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(dogs_vs_cats_config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()


