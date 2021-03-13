# File Name: SVM.py

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from matplotlib.colors import ListedColormap

import warnings

warnings.filterwarnings("ignore")


def gaussian(x, y, sigma=0.1):
    return np.exp(-np.linalg.norm(x - y, axis=1) ** 2 / (2 * (sigma ** 2)))


def polynomial(x, y, b, p=5):
    return (1 + np.dot(x, y) + b) ** p


class SVM(object):
    """
    Support Vector Machine
    """
    """
    ---> from Wikipedia
    
    Any hyperplane can be written as the set of points x satisfying
        w.Tx - b = 0
    where w is the (not necessarily normalized) normal vector to the hyperplane. 
    This is much like Hesse normal form, except that w is not necessarily a unit vector. 
    The parameter {b/||w||} determines the offset of the hyperplane from the origin along the 
    normal vector w.
    
    Hard-margin:
        If the training data is linearly separable, we can select two parallel hyperplanes that separate 
        the two classes of data, so that the distance between them is as large as possible. 
        The region bounded by these two hyperplanes is called the "margin", and the maximum-margin 
        hyperplane is the hyperplane that lies halfway between them. With a normalized or standardized 
        dataset, these hyperplanes can be described by the equations:
            w.Tx - b = 1 (anything on or above this boundary is of one class, with label 1)
        and
            w.Tx - b = -1 (anything on or below this boundary is of the other class, with label −1).
    
        Geometrically, the distance between these two hyperplanes is {2\||w||}, so to maximize the 
        distance between the planes we want to minimize ||w||. 
        The distance is computed using the distance from a point to a plane equation. 
        We also have to prevent data points from falling into the margin, we add the 
        following constraint: for each i either
    
            w.Tx - b >= 1, if y_i = 1, 
        or
            w.Tx - b <= -1, if y_i = -1, 
        These constraints state that each data point must lie on the correct side of the margin.
        This can be rewritten as: 
            y_i(w.Tx_i - b) >= 1, for all 1 <= i <= n

    Soft-margin:
        To extend SVM to cases in which the data are not linearly separable, the hinge loss function is helpful
        --> Hinge Loss = max(0, 1 - y_i(w.Tx_i - b))
    
            l = {   0             if y.f(x) >= 1
                    1 - y.f(x)    otherwise
        The goal of the optimization is to minimize:
            f(w, b) = [1 / N * ∑i=1:n max(0, 1 - y_i(w.Tx_i - b))] + λ||w||^2
        where the parameter λ determines the trade-off between increasing the margin size and ensuring that 
        the x_i lie on the correct side of the margin.
    """

    def __init__(self, alpha=0.1, _lambda=0.01, n_iter=1000):
        """
        Class Constructor
        :param alpha: float (between 0.0 and 1.0)
        :param _lambda: float (between 0.0 and 1.0)
        :param n_iter: int (epochs over the training set)
        """
        # Initialize the parameters
        self.alpha = alpha
        self._lambda = _lambda
        self.n_iter = n_iter

        # Initialize the attributes
        self.weights = None
        self.bias = None

        self.info = []

    def __repr__(self):
        """
        Class Representation:  represent a class's objects as a string
        :return: string
        """
        df = pd.DataFrame.from_dict(self.info)
        pd.set_option('display.max_columns', None)
        df.set_index('Iteration', inplace=True)
        return f'\n ---------- \n Training Model Coefficients \n {df}'

    def fit(self, X, y):
        """
        Fit Method
        :param X: [array-like]
            it is n x m shaped matrix where n is number of samples
                                 and m is number of features
        :param y: [array-like]
            it is n shaped matrix where n is number of samples
        :return: self:object
        """
        # Initialize the parameters
        n_samples, n_features = X.shape
        # Initialize the weights of size n_features with zeros
        self.weights = np.zeros(n_features)
        # Initialize the bias with zero
        self.bias = 0

        temp_dict = {}

        for iteration in range(self.n_iter):
            self.gradient_descent(X, y)

            temp_dict['Iteration'] = iteration
            for i in range(len(self.weights)):
                temp_dict['W' + str(i)] = self.weights[i]
            temp_dict['Bias'] = self.bias
            self.info.append(temp_dict.copy())

    def cost_function(self, x):
        """
        Hyperplane (w.x - b = 0)
        :param x: {array-like}
        :return: {array-like}
        """
        return np.dot(x, self.weights) - self.bias

    def gradient_descent(self, x, y):
        """
        Gradient Descent -- Sub-gradient descent
        :param x: {array_like}
        :param y: {array_like}
        :return: None
        """
        # Loop over each sample in X and y
        for x_i, y_i in zip(x, y):
            # Calculate the partial derivatives for
            # f(w, b) = [1 / N * ∑i=1:n max(0, 1 - y_i(w.Tx_i - b))] + λ||w||^2
            # if y_i(w.Tx_i - b) >= 1, for all 1 <= i <= n
            condition = y_i * (self.cost_function(x_i)) >= 1
            # Sub-gradient descent
            if condition:
                # change in weights = 2λw
                weight_derivative = 2 * self._lambda * self.weights
                # Update Rule: w = w - α . dw
                self.weights -= self.alpha * weight_derivative
            else:
                # change in weights = 2 * λ* w - y_i.x_i
                weight_derivative = 2 * self._lambda * self.weights - np.dot(x_i, y_i)
                # Update Rule: w = w - α * dw
                self.weights -= self.alpha * weight_derivative
                # change in bias = y_i
                bias_derivative = y_i
                # Update Rule: w = w - α * db
                self.bias -= self.alpha * bias_derivative

    def predict(self, x):
        """
        Prediction
        :param x:{array_like}
        :return: {array_like} (0 and 1, or nan)
        """
        model = self.cost_function(x)
        # The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.
        # nan is returned for nan inputs.
        y_predicted = np.sign(model)

        return y_predicted


def plot_svm(svm, X, y):
    """
    Plot the SVM
    :param svm: SVM object
    :param X: {array-like}
    :param y: {array-like}
    :return: None
    """

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    plt.xlabel('X')
    plt.ylabel('y')

    x_min, x_max = np.amin(X[:, 0]), np.amax(X[:, 0])
    y_min, y_max = np.amin(X[:, 1]), np.amax(X[:, 1])

    # ax.set_ylim(y_min - 3, y_max + 3)

    # Step size
    step = 0.9
    print(x_min, x_max)
    print(y_min, y_max)
    x1 = np.arange(x_min, x_max, step)  # np.arange(start, stop, step)
    x2 = np.arange(y_min, y_max, step)

    # Create a mesh to plot: np.meshgrid return coordinate matrices from coordinate vectors.
    xx, yy = np.meshgrid(x1, x2)

    z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    linestyles = ['dashed', 'solid', 'dashed']
    levels = [-0.9, 0.0, 0.9]
    cmap = ListedColormap(['#338BFF', 'red', '#338BFF'])

    # plot contour
    plt.contour(xx, yy, z, levels,
                cmap=cmap,
                linestyles=linestyles)

    # Plot X training points
    plt.scatter(X[:, 0],
                X[:, 1],
                marker='o',
                edgecolor='k',
                c=y,
                cmap=plt.cm.Paired,
                s=30)

    plt.axis('tight')

    plt.show()


if __name__ == '__main__':
    # Create data
    X, y = datasets.make_blobs(n_samples=100,
                               n_features=2,
                               centers=2,
                               cluster_std=1.6,
                               random_state=123
                               )

    # Convert y to reflect -1 and 1 - where the yi are either 1 or −1,
    # each indicating the class to which the point x_i belongs.
    y = np.where(y <= 0, -1, 1)

    start = time.time()

    svm_classifier = SVM(alpha=0.001, _lambda=0.01, n_iter=1000)
    svm_classifier.fit(X, y)
    print(svm_classifier)

    predictions = svm_classifier.predict(X)
    print(f'Accuracy is: {np.sum(y == predictions) / len(y)}')

    end = time.time()  # ----------------------------------------------
    print('\n ----------\n Execution Time: {%f}' % ((end - start) / 1000) + ' seconds.')

    plot_svm(svm_classifier, X, y)