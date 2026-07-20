import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def mnist(n):
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    datax = digits.images.reshape((n_samples, -1))
    datax = StandardScaler().fit_transform(datax)
    return datax.astype(np.float32)[:n], digits.target[:n]
