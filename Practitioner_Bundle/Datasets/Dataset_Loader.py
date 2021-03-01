import os
import cv2
import numpy as np

class DatasetLoader:
    def __init__(self, preprocessors = None):
        #store the preprocessor
        self.preprocessors = preprocessors

    def load(self, imagedir_path, verbose=-1):
        data   = []
        labels = []
        image_paths = []

        for dir, folders, files in os.walk(imagedir_path):
            for image_file in files:
                image_paths.append(os.path.join(dir, image_file))

            for (i, image_path) in enumerate(image_paths):
                image = cv2.imread(image_path)
                label = image_path.split(os.path.sep)[-2]

                if self.preprocessors is not None:
                    for preprocessor_type in self.preprocessors:
                        image = preprocessor_type(image)

                data.append(image)
                labels.append(label)            

                if verbose > 0 and i > 0 and (i+1)%verbose == 0:
                    print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))
        
        return (np.array(data), np.array(labels))  

