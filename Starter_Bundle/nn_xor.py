from models.neuralnetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 2, 1], alpha=0.5)
losses = nn.fit(X, y, epochs=12000)


for (x, target) in zip(X, y):
    # make a prediction on the data point and display the result to our console
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target[0], pred, step))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 12000), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()    