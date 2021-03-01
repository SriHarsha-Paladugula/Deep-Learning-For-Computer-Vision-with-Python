import os
import cv2
from keras.preprocessing.image import img_to_array
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np

class Preprocess:
    def __init__(self, width, height, inter = cv2.INTER_AREA):
        self.width  = width
        self.height = height
        self.inter  = inter

    def image_to_array(self, image, dataFormat=None):
        # apply the Keras utility function that correctly rearranges the dimensions of the image
        return img_to_array(image, data_format = dataFormat)


    def resize(self, image):
        # Resize the image to a fixed size, ignoring the aspect ratio
        resized_image = cv2.resize(image, (self.width, self.height), interpolation=self.inter)    
        return resized_image

    def aspect_aware_resize(self, image):
        #To get the height and width of the image
        H,W = image.shape[:2]
        dH = 0
        dW = 0

        #Get the smallest dimension of the two and resize along that dimension and determine the delta offsets
        # we will be using when cropping along the longer dimension

        if W < H:
            # calculate the ratio of heights to construct the dimensions
            r = self.width/float(W)
            dim = (W, int(H*r))
            image = cv2.resize(image, dim, interpolation=self.inter)
            dH = int((image.shape[0] - self.height)/2.0)
        else:
            r = self.height/float(H)
            dim = (int(W*r), H)
            image = cv2.resize(image, dim, interpolation=self.inter)
            dW = int((image.shape[1] - self.width)/2.0)
        
        H, W = image.shape[:2]
        image = image[dH: H-dH, dW: W-dW]

        return cv2.resize(image, (self.width,self.height), interpolation=self.inter)  

    def Mean_preprocessor(self, image, rMean, gMean, bMean):
        # store the Red, Green, and Blue channel averages across a training set
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean          
        
        # split the image into its respective Red, Green, and Blue channels
        (B, G, R) = cv2.split(image.astype("float32"))

        # subtract the means for each channel
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        # merge the channels back together and return the image
        return cv2.merge([B, G, R])

    def Patch_preprocessor(self, image):
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]

    def Crop_preprocessor(self, image, horiz=True):
        # initialize the list of crops
        crops = []

        # grab the width and height of the image then use these dimensions to define the corners of the image based
        (h, w) = image.shape[:2]
        coords = [[0, 0, self.width, self.height],
                [w - self.width, 0, w, self.height],
                [w - self.width, h - self.height, w, h],
                [0, h - self.height, self.width, h]]

        # compute the center crop of the image as well
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW, dH, w - dW, h - dH])

        # loop over the coordinates, extract each of the crops, and resize each of them to a fixed size
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height), interpolation=self.inter)
            crops.append(crop)

        # check to see if the horizontal flips should be taken
        if self.horiz:
            # compute the horizontal mirror flips for each crop
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)       
        
        # return the set of crops
        return np.array(crops)


