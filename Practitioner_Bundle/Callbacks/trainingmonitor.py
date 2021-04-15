# import the necessary packages

from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath = None, startAt = 0):
        #store the output path for the figure, the path to the JSON serialized file, 
        #and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath  = figPath
        self.jsonPath = jsonPath
        self.startAt  = startAt

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.History = {}

        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.History = json.loads(open(self.jsonPath).read())

                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    #loop over the entries in the history log and trim any entries that are past 
                    #the starting epoch
                    for key in self.History.keys():
                        self.History[key] = self.History[key][:self.startAt]    

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc. for the entire training process
        for (key, value) in logs.items():
            loss = self.History.get(key, [])
            loss.append(value)
            self.History[key] = loss

        # check to see if the training history should be serialized to file 
        if self.jsonPath is not None:
            f = open(self.jsonPath, 'w')
            f.write(json.dumps(self.History))
            f.close()

        # ensure at least two epochs have passed before plotting (epoch starts at zero)
        if len(self.History["loss"]) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.History["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.History["loss"], label = "train_losss")
            plt.plot(N, self.History["val_loss"], label = "val_loss")
            plt.plot(N, self.History["accuracy"], label = "train_acc")
            plt.plot(N, self.History["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.History["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # save the figure
            plt.savefig(self.figPath)
            plt.close()       


