import h5py
import os


class HDF5DatasetWriter:
    def __init__(self, dims, outputpath, datakeys='images', buffsize=1000):
        # check to see if the output path exists, and if so, raise an exception
        if os.path.exists(outputpath):
            raise ValueError("The supplied ‘outputPath‘ already exists and cannot be overwritten. Manually delete the "
                             "file before continuing.", outputpath)

        # open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another to store the class labels

        self.db = h5py.File(outputpath, "w")
        self.data = self.db.create_dataset(datakeys, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0], ), dtype="int")

        # store the buffer size, then initialize the buffer itself along with the index into the datasets

        self.bufferSize = buffsize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        # add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) > self.bufferSize:
            self.flush()

    def flush(self):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx: i] = self.buffer["data"]
        self.labels[self.idx: i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def store_class_labels(self, class_labels):
        # create a dataset to store the actual class label names, then store the class labels
        dt = h5py.special_dtype(vlen=str)
        labelset = self.db.create_dataset("label_names", (len(class_labels), ), dtype=dt)
        labelset[:] = class_labels

    def close(self):
        # check to see if there are any other entries in the buffer that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset

        self.db.close()