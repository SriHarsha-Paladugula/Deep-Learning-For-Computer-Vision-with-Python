# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")


from callbacks.trainingmonitor import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from models.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True, help = "path to the output directory")
args = vars(ap.parse_args())

# show information on the process ID
print("[INFO process ID: {}".format(os.getpid()))

# load the training and testing data, then scale it into the range [0, 1]
print("[INFO] loading CIFAR-10 data...")

((train_X, train_y), (test_X, test_y)) = cifar10.load_data()
train_X = train_X.astype("float")/255.0
test_X  = test_X.astype("float")/255.0

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y  = lb.transform(test_y)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# initialize the SGD optimizer, but without any learning rate decay
print("[INFO] compiling model...")
opt = SGD(lr=0.01, momentum = 0.9, nesterov = True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ['accuracy'])

# construct the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

# train the network
print("[INFO] training network...")
model.fit(train_X, train_y, validation_data=(test_X, test_y), batch_size=64, epochs=100, 
callbacks=callbacks, verbose=1)