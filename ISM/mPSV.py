import numpy as np

class multiPSV_chi2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def PSV(self, A, x0, gammaG, gammaL):
        fG = 2 * gammaG * 1.17741
        fL = 2 * gammaL
        f = np.power(np.power(fG, 5) + 2.69269 * np.power(fG, 4) * fL + 2.42843 * np.power(fG, 3)\
                     * np.power(fL,2) + 4.47163 * np.power(fG, 2) * np.power(fL, 3)\
                     + 0.07842 * fG * np.power(fL, 4) + np.power(fL, 5), 1. / 5)
        eta = 1.36603 * (fL / f) - 0.47719 * np.power(fL / f, 2) + 0.11116 * np.power(fL / f, 3)
        fcn = eta * np.exp(-0.5 * np.power((self.x - x0) / gammaG, 2)) / (np.sqrt(np.pi) * gammaG)
        fcn += (1 - eta) * np.power(1 + np.power((self.x - x0) / gammaL, 2), -1) / (np.pi * gammaL)
        fcn *= A * 10000
        return fcn

    def multiPSV(self, *args):
        f = 0.
        for s in range(int(len(args) / 4)):
            f += self.PSV(args[s * 4 + 0], args[s * 4 + 1], args[s * 4 + 2], args[s * 4 + 3])
        return f

    def __call__(self, *args):
        chi2 = np.sum(1.e-2 * np.power(self.y - self.multiPSV(*args), 2))
        return chi2