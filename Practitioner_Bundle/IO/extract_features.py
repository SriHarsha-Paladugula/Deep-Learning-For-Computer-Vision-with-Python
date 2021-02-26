from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from IO.hdf5datasetwriter import HDF5DatasetWriter
import numpy as np
import progressbar
import argparse
import random
import os
import warnings
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
ap.add_argument("-o", "--output_path", required=True, help="Path to output HDF5 File")
ap.add_argument("-b", "--batch_size", type=int, default=32, help="batch size of images to be passed through network")
ap.add_argument("-s", "--buffer_size", type=int, default=1000, help="size of feature extraction buffer")
args = vars(ap.parse_args())

# store the batch size in a convenience variable
bs = args["batch_size"]

# grab the list of images that we’ll be describing then randomly shuffle them to allow for easy training and testing
# splits via array slicing during training time

print("[INFO] loading images...")
image_paths = []

for subdir, dirs, files in os.walk(args["dataset"]):
    image_paths += [os.path.join(subdir, image_file) for image_file in files]

random.shuffle(image_paths)

# extract the class labels from the image paths then encode the labels

labels = [p.split(os.path.sep)[-2] for p in image_paths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load the VGG16 network
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)

# initialize the HDF5 dataset writer, then store the class label names in the dataset

dataset = HDF5DatasetWriter((len(image_paths), 7*7*512), args["output_path"], datakeys="features", buffsize=bs)

dataset.store_class_labels(le.classes_)

# initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(len(image_paths), widgets).start()

# loop over the images in batches

for i in range(0, len(image_paths), bs):
    # extract the batch of images and labels, then initialize the list of actual images that will be passed through
    # the network for feature extraction
    batch_paths = image_paths[i:i+bs]
    batch_labels = labels[i:i+bs]
    batch_Images = []
    # loop over the images and labels in the current batch
    for (j, image_path) in enumerate(batch_paths):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess the image by (1) expanding the dimensions and (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # add the image to the batch
        batch_Images.append(image)

    # pass the images through the network and use the outputs as our actual features
    batch_Images = np.vstack(batch_Images)
    features = model.predict(batch_Images, batch_size=bs)

    # reshape the features so that each image is represented by a flattened feature vector of the ‘MaxPooling2D‘ outputs
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # add the features and labels to our HDF5 dataset
    dataset.add(features, batch_labels)
    pbar.update(i)

# close the dataset
dataset.close()
pbar.finish()
