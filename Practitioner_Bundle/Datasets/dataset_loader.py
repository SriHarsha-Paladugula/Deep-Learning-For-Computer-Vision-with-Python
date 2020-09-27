import numpy as np
import os
import cv2

class SimpleDatasetLoader:
    def __init__(self, preprocessors =None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagedir_path, verbose=-1):
        
        data = []
        labels = []
        imagePaths =[]

        for r, _, f in os.walk(imagedir_path):
            for file in f:
                    imagePaths.append(os.path.join(r, file))
        
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for preprocessor_type in self.preprocessors:
                    image = preprocessor_type.preprocess(image)

            data.append(image)
            labels.append(label)

            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

        return (np.array(data), np.array(labels))                    



