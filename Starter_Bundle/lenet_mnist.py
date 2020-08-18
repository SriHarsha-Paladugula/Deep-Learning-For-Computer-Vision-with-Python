from models.lenet import LeNet
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K


# grab the MNIST dataset (if this is your first time using this dataset then the 55MB 
# download may take a minute)
print("[INFO] accessing MNIST...")
X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

if K.image_data_format() == "channels_first":
    X = X.reshape(X.shape[0], 1, 28, 28)

# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
    X = X.reshape(X.shape[0], 28, 28, 1)

X = X.astype("float") / 255.0

lb = LabelBinarizer()
y  = lb.fit_transform(y)    

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.25, random_state=42)

print("[INFO] compiling model...")
opt = SGD(0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss='categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=20, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()