"""First version of RNN using numpy, hardcoded to use one hidden layer. All the equations are from
http://www.deeplearningbook.org/contents/rnn.html"""

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
        self.biases = np.array([self.b, self.c])
        self.weights = np.array([self.U, self.W, self.V])
        self.params = np.array([self.b, self.c, self.U, self.W, self.V])

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

    def backprop(self, xs, ys, h_prev=None, trunc=None):
        """
        Backprop through time to compute gradients of softmax loss function wrt parameters, using one training example;
        this is just standard backprop applied to the unrolled computational graph of RNN. No side effects.

        :param xs: input data sequence codes (iterable), e.g. [2, 0, 13, 11]. Each input code should be in
        range(self.data_size); indexing self.W (embedding matrix) by an input code is equivalent to multiplying it with
        a one-hot input vector
        :param ys: target output data sequence codes (iterable), e.g. [0, 13, 11, 5]
        :param h_prev: hidden unit at time t=-1
        :param trunc: number of back propagation steps to perform; default to len(xs), i.e. do BPTT exactly; else if
        trunc < len(xs), BPTT will only be done on the last `trunc` elements, so the gradients will be approximate
        :return: tuple of gradients of loss function wrt model parameters, and loss: (array(db, dc, dU, dW, dV), loss)
        """
        T = len(xs)  # training sequence length
        assert T == len(ys)

        if h_prev is None:  # use the RNN's current hidden unit by default
            h_prev = self.h
            h = self.h
        else:
            h = h_prev

        if trunc is None or trunc > T:  # perform BPTT exactly, on the entire sequence
            trunc = T
        skip = T - trunc  # the number of BPTT steps to skip

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

        # BPTT; only perform (T-truncation) time steps of computations
        dhs = np.empty_like(hs)
        dhs[-1] = np.dot(self.V.T, dos[-1])  # (10.19) for last time step

        for t in range(T - 2, skip - 1, -1):
            dhs[t] = np.dot(self.W.T, (1 - hs[t + 1] ** 2) * dhs[t + 1]) + np.dot(self.V.T, dos[t])  # (10.21)

        dc = np.sum(dos[skip:], axis=0)  # (10.22)
        db = np.sum((1 - hs[skip:] ** 2) * dhs[skip:], axis=0)  # (10.23)
        dV = np.dot(dos[skip:].T, hs[skip:])  # more efficient than the sum of outer products in (10.24)
        if skip == 0:  # in case we BPTT all the way to the beginning
            dW = np.dot(((1 - hs[skip:] ** 2) * dhs[skip:]).T,
                        np.vstack((h_prev, hs))[:T - skip])  # (10.26)
        else:
            dW = np.dot(((1 - hs[skip:] ** 2) * dhs[skip:]).T, hs[skip - 1:-1])  # (10.26)
        dU = np.dot(((1 - hs[skip:] ** 2) * dhs[skip:]).T,
                    np.array(onehot(xs[skip:], self.data_size)))  # (10.28)

        return np.array([db, dc, dU, dW, dV]), L

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

    def sgd(self, X, Y, trunc=None, eta=0.05, adagrad=True, num_epochs=100, report_interval=25, element_map=None):
        """
        Train with SGD given training and target sequences
        :param X: list of training sequences, something like [[1,3,0,29,8],[17,8,2],...]
        :param Y: list of target sequences, like [[,3,0,29,8,6],[8,2,0],...]
        :param trunc: max number of BPTT to perform for a given example
        :param eta: learning rate; this will be constant if adagrad=False
        :param adagrad: whether to adapt learning rate using AdaGrad
        :param num_epochs:
        :param report_interval: report progress every such number of epochs
        :param element_map: dictionary that maps element id (int) to the object it represents, e.g. {0:'a', 1:'b', ...};
         optional, if provided, will sample the RNN to generate a sequence of such objects and print it
        """
        assert len(X) == len(Y)
        N = len(X)
        if adagrad:
            grad_hist = np.array([np.zeros_like(p) for p in self.params])

        idx = np.arange(N)

        for n in range(num_epochs):
            np.random.shuffle(idx)
            L = 0
            for i in idx:
                # do one SGD step
                dparams, L_i = self.backprop(X[i], Y[i], trunc)
                for dparam in dparams:
                    np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
                if adagrad:
                    grad_hist += np.square(dparams)
                    param_updates = -eta * dparams / (grad_hist ** 0.5 + 1e-7)
                else:
                    param_updates = -eta * dparams

                for p, pu in zip(self.params, param_updates):   # note `self.params[:]+=param_updates` won't work
                    p[:] += pu

                L += L_i / len(X[i])
            # sample from the model now and then
            if n % report_interval == 0:
                print('epoch %d, loss %f' % (n, L))
                sample_ix = self.sample(200)
                txt = ''.join(element_map[ix] for ix in sample_ix)
                print('----\n %s \n----' % (txt,))

    def sample(self, T, seed_x=None, h_prev=None):
        """
        Sample a sequence of integers from the model
        :param T: length of sequence to sample
        :param seed_x: initial input element
        :param h_prev: previous hidden state
        :return: a sampled sequence
        """
        if h_prev is None:  # use the RNN's current hidden unit by default
            h = self.h
        else:
            h = h_prev
        if seed_x is None:
            seed_x = np.random.randint(0, self.data_size)
        x = seed_x
        xs = [x]
        for t in range(T - 1):
            a = self.b + self.U[:, x] + np.dot(self.W,
                                               h)  # (10.8); note that indexing by xs[t] is equiv to multiplication
            h = np.tanh(a)  # (10.9)
            o = self.c + np.dot(self.V, h)  # (10.10)
            p = softmax(o)  # (10.11), conditional dist for the next char
            x = np.random.choice(self.data_size, p=p.ravel())
            xs.append(x)

        return xs
