"""First version of RNN using numpy, hardcoded to use one hidden layer"""

import numpy as np

from util import softmax, onehot


class RNNNumpy:
    def __init__(self, data_size, hidden_size):
        """
        :param data_size: size of a single input/output element at each time (we assume they have the same length);
        for example, if we're modeling ASCII characters, then this size might be 256, with an input seq like [1,255,4]
        :param hidden_size: length of hidden unit vec
        """
        self.data_size, self.hidden_size = data_size, hidden_size
        self.h = np.empty(hidden_size)  # hidden unit (not a parameter)
        self.b = np.empty(hidden_size)  # hidden bias
        self.c = np.empty(data_size)  # output bias
        self.U = np.empty((hidden_size, data_size))  # input to hidden weights
        self.W = np.empty((hidden_size, hidden_size))  # hidden to hidden weights
        self.V = np.empty((data_size, hidden_size))  # hidden to output weights
        # for convenience
        self.biases = (self.b, self.c)
        self.weights = (self.U, self.W, self.V)
        self.params = self.biases + self.weights

    def init_params(self):
        """Initialize params"""
        for w in self.weights:  # initialize weights with standard Gaussian with std 1/sqrt(# incoming connections)
            # p.shape[-1] is the number of connections from previous layer
            w[:] = np.random.rand(*w.shape) / np.sqrt(w.shape[-1])
        for b in self.biases:  # init biases to zeros
            b[:] = np.zeros_like(b)

    def init_h(self, h0=None):
        """Initialize the hidden state vector; zero by default"""
        if h0 is None:
            # h0 = np.random.rand(*self.h.shape)
            h0 = np.zeros_like(self.h)
        self.h = h0

    # def step(self, x):
    #     """
    #     Perform one time step of computation
    #     :param x: input data vec (one element of sequence)
    #     :return: output vec
    #     """
    #     a = self.b + np.dot(self.U, x) + np.dot(self.W, self.h)  # (10.8)
    #     h = np.tanh(a)  # (10.9)
    #     o = self.c + np.dot(self.V, h)  # (10.10)
    #     self.h = h
    #     return softmax(o)  # (10.11)
    #
    # def predict(self, xs):
    #     """
    #     Feed forward computation across time given a sequence of inputs to get probability outputs
    #     :param xs: input data sequence (iterable)
    #     :return: a list of output data probabilities at each time step
    #     """
    #     y_hats = []
    #     for x in xs:
    #         y_hats.append(self.step(x))
    #     return y_hats



    def backprop(self, xs, ys, h_prev=None, truncation=0):
        """
        Backprop through time to compute gradients of softmax loss function wrt parameters, using one training example;
        this is just standard backprop applied to the unrolled computational graph of RNN. No side effects.

        :param xs: input data sequence codes (iterable), e.g. [2, 0, 13, 11]. Each input code should be in
        range(self.data_size); indexing self.W (embedding matrix) by an input code is equivalent to multiplying it with
        a one-hot input vector
        :param ys: target output data sequence codes (iterable), e.g. [0, 13, 11, 5]
        :param h_prev: hidden unit at time t=-1
        :param truncation: number of BPTT steps to skip; 0 by default, i.e. full BPTT for the training example
        :return: tuple of gradients of loss function wrt model parameters, and loss: ((db, dc, dU, dW, dV), loss)
        """
        T = len(xs)  # training sequence length
        assert T == len(ys)

        if h_prev is None:  # use the RNN's current hidden unit by default
            h_prev = self.h
            h = self.h
        else:
            h = h_prev

        # feedforward to get predictions/intermediate variables useful for backprop
        hs = np.empty((T, self.hidden_size))  # hidden states across time
        os = np.empty((T, self.data_size))  # class scores (args to softmax)
        y_hats = np.empty_like(os)  # predicted probs

        L = 0
        for t in range(T):
            a = self.b + self.U[:, xs[t]] + np.dot(self.W,
                                                   h)  # (10.8); note that indexing by xs[t] is equiv to multiplication
            h = np.tanh(a)  # (10.9)
            o = self.c + np.dot(self.V, h)  # (10.10)
            y_hat = softmax(o)  # (10.11)
            L += -np.log(y_hat[ys[t]])

            hs[t] = h
            os[t] = o
            y_hats[t] = y_hat

        dos = np.copy(y_hats)
        dos[np.arange(T), ys] -= 1  # (10.18), for all time steps

        # BPTT; only perform T-truncation time steps of computations
        dhs = np.empty_like(hs)
        dhs[-1] = np.dot(self.V.T, dos[-1])  # (10.19) for last time step

        for t in range(T - 2, truncation - 1, -1):
            dhs[t] = np.dot(self.W.T, (1 - hs[t + 1] ** 2) * dhs[t + 1]) + np.dot(self.V.T, dos[t])  # (10.21)

        dc = np.sum(dos[truncation:], axis=0)  # (10.22)
        db = np.sum((1 - hs[truncation:] ** 2) * dhs[truncation:], axis=0)  # (10.23)
        dV = np.dot(dos[truncation:].T, hs[truncation:])  # more efficient than the sum of outer products in (10.24)
        if truncation == 0:  # in case we BPTT all the way to the beginning
            dW = np.dot(((1 - hs[truncation:] ** 2) * dhs[truncation:]).T,
                        np.vstack((h_prev, hs))[:T - truncation])  # (10.26)
        else:
            dW = np.dot(((1 - hs[truncation:] ** 2) * dhs[truncation:]).T, hs[truncation - 1:-1])  # (10.26)
        dU = np.dot(((1 - hs[truncation:] ** 2) * dhs[truncation:]).T,
                    np.array(onehot(xs[truncation:], self.data_size)))  # (10.28)

        return (db, dc, dU, dW, dV), L

    def loss(self, xs, ys, h_prev=None):
        """
        Feed forward to calculate the softmax loss on a input/target pair of sequence
        target sequence is provided. No side effects.
        :param xs: input sequence
        :param ys: target sequence
        :param h_prev: hidden unit at time t=-1
        :return: L
        """
        T = len(xs)
        if h_prev is None:  # use the RNN's current hidden unit by default
            h = self.h
        else:
            h = h_prev
        L = 0

        for t in range(T):
            a = self.b + self.U[:, xs[t]] + np.dot(self.W,
                                                   h)  # (10.8); note that indexing by xs[t] is equiv to multiplication
            h = np.tanh(a)  # (10.9)
            o = self.c + np.dot(self.V, h)  # (10.10)
            y_hat = softmax(o)  # (10.11)

            if ys is not None:
                L += -np.log(y_hat[ys[t]])

        return L
