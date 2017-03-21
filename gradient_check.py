import numpy as np

from RNNNumpy import RNNNumpy

rnn = RNNNumpy(5, 3)

xs = np.array([0, 1, 2, 3])
ys = np.array([1, 2, 3, 0])

rnn.init_h()  # self.h is fixed to zero vec the entire time
rnn.init_params()

h, threshold = 1e-4, 1e-6
dparams, _ = rnn.backprop(xs, ys)
for dparam, param, param_name in zip(dparams, rnn.params, ['b', 'c', 'U', 'W', 'V']):
    assert dparam.shape == param.shape
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:  # check partial derivatives wrt each param
        ix = it.multi_index
        old_val = param[ix]
        param[ix] = old_val + h
        L_plus = rnn.loss(xs, ys)  # L(param+h)
        param[ix] = old_val - h
        L_minus = rnn.loss(xs, ys)  # L(param-h)
        param[ix] = old_val
        num_grad = (L_plus - L_minus) / (2 * h)
        bp_grad = dparam[ix]
        if num_grad == bp_grad == 0:  # to avoid division by zero below
            rel_err = 0
        else:
            rel_err = abs(num_grad - bp_grad) / max(abs(num_grad), abs(bp_grad))  # relative error
        # print(rel_err)
        if rel_err > threshold:
            print("Gradient Check ERROR: parameter=%s ix=%s" % (param_name, ix))
            print("Numerical gradient: %f" % num_grad)
            print("Backpropagation gradient: %f" % bp_grad)
            print("Relative Error: %f" % rel_err)
        it.iternext()
