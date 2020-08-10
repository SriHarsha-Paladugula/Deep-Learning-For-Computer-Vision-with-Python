import numpy as np


class NeuralNetwork:

    def __init__(self, layers, alpha=0.1):

        self.weights = []
        self.alpha   = alpha
        self.layers  = layers

        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.weights.append(w / np.sqrt(layers[i]))
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.weights.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represents the networkarchitecture
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))    
   
    def sigmoid(self, x):
        # compute and return the sigmoid activation value for a given input value
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # compute the derivative of the sigmoid function ASSUMING that ‘x‘ has already been passed 
        # through the ‘sigmoid‘ function
        return x * (1 - x)

    def fit_partial(self, x, y):
        # construct our list of output activations for each layer as our data point flows through 
        # the network; the first activation is a special case -- it’s just the input feature vector itself
        A = [np.atleast_2d(x)]

        for layer in np.arange(0, len(self.weights)):

            net = A[layer].dot(self.weights[layer])
            out = self.sigmoid(net)
            A.append(out)

            #BACKPROPAGATION
            #The first phase of backpropagation is to compute the difference between our *prediction* 
            #(the final output activation in the activations list) and the true target value 

        error = A[-1] - y
        D = [error * self.sigmoid_deriv(A[-1])] 

        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.weights[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        D = D[::-1]
        # WEIGHT UPDATE PHASE
        # loop over the layers
        for layer in np.arange(0, len(self.weights)):
            self.weights[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def calculate_loss(self, X, targets):
        # make predictions for the input data points then compute the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss              

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        losses = []
        X = np.c_[X, np.ones((X.shape[0]))]
        for epoch in np.arange(0, epochs):
            # loop over each individual data point and train our network on it
            loss = []
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
                # check to see if we should display a training update
                loss.append(self.calculate_loss(X, y))
            avg_loss = np.average(loss)
            losses.append(avg_loss)
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:    
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, avg_loss))
        return losses            

    def predict(self, X, addBias=True):

        p = np.atleast_2d(X)
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))] 
        for layer in np.arange(0, len(self.weights)):
            p = self.sigmoid(np.dot(p, self.weights[layer]))

        return p                               