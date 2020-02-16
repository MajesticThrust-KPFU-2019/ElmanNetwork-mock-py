import numpy as np
import random
from typing import List


class Network:
    def __init__(self, sizes):
        """
        Arguments:
            shape - an array of layer sizes. For example, [5,4,3] - a net
            with 5 input neurons, 1 layer of 4 hidden neurons, and 3 output
            neurons
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        # bias is essentially a weight for a fixed input of 1
        # include biases in weights
        self.weights = [
            # np.random.rand(y + 1, x + 1) for x, y in zip(sizes[:-1], sizes[1:])
            np.random.rand(y, x) for x, y in zip(sizes[:-1], sizes[1:])
        ]

    def feedforward(self, x):
        """Return the output of the network if a` is input."""
        # x = np.resize(x, [self.sizes[0] + 1, 1])
        x = np.array(x)
        x.resize((self.sizes[0], 1))

        for w in self.weights:
            x = sigmoid(np.matmul(w, x))

        return x

    def gradient_descent(
        self, training_data, epochs, mini_batch_size, learning_rate, test_data=None
    ):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if test_data:
                # print(
                #     "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
                # )
                print(f"Epoch {j}: loss = {self.evaluate(test_data)}")
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``learning_rate``
        is the learning rate."""
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_w = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # change weights by averaged gradient delta
        self.weights = [
            w - (learning_rate / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.weights``."""

        # gradients of the weights, for each layer
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward

        y = np.array(y)
        y.resize((self.sizes[-1], 1))

        activation = np.array(x)
        activation.resize((self.sizes[0], 1))

        # list to store all the activations, layer by layer
        activations = [activation]

        # list to store all the z vectors, layer by layer
        zs = []

        for w in self.weights:
            # z = np.dot(w, activation)
            z = np.matmul(w, activation)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass

        # print(f"Activation -1:\n{activations[-1]}, shape: {activations[-1].shape}")
        # print(f"Y:\n{y}, shape: {y.shape}")
        # print(f"Z -1:\n{zs[-1]}, shape: {zs[-1].shape}")
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # print(f"Delta:\n{delta}, shape: {delta.shape}")
        # print(f"Activation -2: \n{activations[-2].transpose()}, shape: {activations[-2].transpose().shape}")
        nabla_w[-1] = np.matmul(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)

            # sigmoid_prime multiplied elementwise?
            # weights multiplied by delta as matrices?
            # why this?
            delta = sp * np.matmul(self.weights[-l + 1].transpose(), delta)

            # print(f"Delta: {delta}")
            # print(f"Activation: {activations[-l].transpose()}, shape: {activations[-l].transpose().shape}")
            # print(f"Activation - 1: {activations[-l - 1].transpose()}, shape: {activations[-l - 1].transpose()}")

            # the gradient of the weights on the current layer
            nabla_w[-l] = np.matmul(delta, activations[-l - 1].transpose())
            # nabla_w[-l] = np.matmul(delta, activations[-l].transpose())

        return nabla_w

    def evaluate(self, test_data):
        """Return the value of the loss function"""
        # """Return the number of test inputs for which the neural
        # network outputs the correct result. Note that the neural
        # network's output is assumed to be the index of whichever
        # neuron in the final layer has the highest activation."""
        # test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        # return sum(int(x == y) for (x, y) in test_results)

        test_results = [(self.feedforward(x), y) for (x, y) in test_data]
        loss = sum(self.cost(x, y) for (x, y) in test_results) / len(test_results)
        return loss

    def cost(self, output_activations, y):
        y = np.array(y)
        y.resize((self.sizes[-1], 1))
        # assert len(output_activations) == len(y)
        # print(f"Output:\n{output_activations}, shape: {output_activations.shape}")
        # print(f"Y:\n{y}, shape: {y.shape}")
        cost = sum((output_activations - y)**2) / len(y)
        # print(f"Cost: {cost}, shape: {cost.shape}")
        return float(cost)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations - y


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    # elementwise multiplication
    return sigmoid(z) * (1 - sigmoid(z))

