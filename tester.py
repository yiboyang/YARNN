import numpy as np

from RNNNumpy import RNNNumpy

rnn = RNNNumpy(5, 3)

xs = np.array([0, 2])
ys = np.array([2, 3])

rnn.init_h()
rnn.init_params()
print(rnn.backprop(xs, ys))
