import os
import cv2
from keras.preprocessing.image import img_to_array

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
