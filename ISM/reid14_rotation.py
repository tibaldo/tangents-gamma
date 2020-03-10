import numpy as np

##Universal roation curve Persic1996
##with parameters for the MW from Reid2014

R0 = 8.31
V0 = 241
a1 = 241.
a2 = 0.9
a3 = 1.46

A = 1.97 * a2 ** -1.22 / (a2 ** -2 + 0.78 ** 2) ** 1.43
B = (1 + a3 ** 2) * a2 ** -2 / (a2 ** -2 + a3 ** 2)
beta = (V0 ** 2 / a1 ** 2 - B) / (A - B)


def rotcurve(R):
    x = R / (a2 * R0)
    val = beta * 1.97 * x ** 1.22 / (x ** 2 + 0.78 ** 2) ** 1.43
    val += (1 - beta) * (1 + a3 ** 2) * x ** 2 / (x ** 2 + a3 ** 2)
    val *= a1 ** 2
    val = np.sqrt(val)
    return val