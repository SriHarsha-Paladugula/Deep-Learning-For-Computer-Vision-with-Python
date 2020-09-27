from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from Preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from Preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from Datasets.dataset_loader import SimpleDatasetLoader
from models.minivggnet import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os