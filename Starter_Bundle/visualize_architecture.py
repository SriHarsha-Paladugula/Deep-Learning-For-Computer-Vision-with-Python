# import the necessary packages
from models.minivggnet import MiniVGGNet
from keras.utils import plot_model

# initialize LeNet and then write the network architecture visualization graph to disk
model = MiniVGGNet.build(32, 32, 3, 10)
plot_model(model, to_file="MiniVGGNet.png", show_shapes=True)