import numpy as np

class Perceptron:

    def __init__(self, N, alpha = 0.1):

        self.weights = np.random.randn(N+1)/np.sqrt(N)
        self.alpha   = alpha

    def step(self, x):

        return 1 if x > 0 else 0

    def fit(self, X, y, epochs = 10):

        # insert a column of 1’s as the last entry in the feature  matrix -- this little trick allows us to treat the bias
        # as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                prediction = self.step(np.dot(x, self.weights))

                if prediction != target:
                    error = prediction - target
                    self.weights += -self.alpha * error * x

    def predict(self, X, addBias=True):

        X = np.atleast_2d(X)
        if addBias:
            # insert a column of 1’s as the last entry in the feature matrix (bias)
            X = np.c_[X, np.ones((X.shape[0]))]

        return self.step(np.dot(X, self.weights))    
