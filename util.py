"""Miscellaneous functions/helpers"""
import numpy as np


def softmax(z):
    z -= z.max()
    xp = np.exp(z)
    return xp / xp.sum()


def onehot(z, k):
    """
    Given a vector of ints, return a matrix of vectors in one-of-k encoding
    e.g. one_hot([0,1],3) -> np.array([[1,0,0],[0,1,0]])
    """
    m = np.zeros((len(z), k), dtype=float)
    for i in range(len(z)):
        m[i][z[i]] = 1
    return m
