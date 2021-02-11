import os
import numpy as np
import matplotlib.pyplot as plt
from Models.minivggnet import MiniVGGNet
from Datasets.Dataset_Loader import DatasetLoader
from Preprocessing.Preprocessor import Preprocess
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset", default="./Datasets/17flowers", help="Path to the dataset")
args = vars(ap.parse_args())


# load the dataset from disk then scale the raw pixel intensities to the range [0, 1]
aug = Preprocess(64,64)
sdl = DatasetLoader(preprocessors = [aug.image_to_array, aug.aspect_aware_resize])
(data, labels) = sdl.load(args['dataset'])


# to get the total number of classes being labels in the dataset
no_of_classes = len(np.unique(labels))
print("Total No of classes in the classification task is :", no_of_classes)

data = data.astype("float")/255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY  = LabelBinarizer().fit_transform(testY)


# initialize the optimizer and model
print("[INFO] compiling the model ....")
opt = SGD(lr=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=no_of_classes)
model.compile(optimizer=opt, metrics=['accuracy'], loss='categorical_crossentropy')

# Training the Network
print("[INFO] training network...")
H = model.fit(trainX, trainY, batch_size=32, epochs=100, validation_data=(testX, testY), shuffle=True, verbose=1)

# Evaluating the network
predictions = model.predict(testX)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))

# plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,100), H.history['loss'], label = "train_loss")
plt.plot(np.arange(0,100), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0,100), H.history['accuracy'], label="train_accuracy")
plt.plot(np.arange(0,100), H.history['val_accuracy'], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("loss/Accuracy")
plt.legend()
plt.show()