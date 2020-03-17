import numpy as np

##Universal roation curve Persic1996
##with parameters for the MW from Reid2019, fit A5 in Table 3
##translated from Fortran code in appendix B

R0 = 8.15
V0 = 236.
a2 = 0.96
a3 = 1.62

# pre-compute some quantities independent of R

lbd = np.power(a3 / 1.5, 5)  # this is L/L*
log_lbd = np.log10(lbd)
Ropt = a2 * R0

term1 = 200. * np.power(lbd, 0.41)

top = 0.75 * np.exp(-0.4 * lbd)
bot = 0.47 + 2.25 * np.power(lbd, 0.4)
term2 = np.sqrt(0.8 + 0.49 * log_lbd + top / bot)


def rotcurve(R):

    rho = R / Ropt

    top = 1.97 * np.power(rho,1.22)
    bot = np.power(rho**2 + 0.61,1.43)
    term3 = (0.72 + 0.44 * log_lbd) * top/bot

    top = rho**2
    bot = rho**2 + 2.25 * np.power(lbd,0.4)
    term4 = 1.6 * np.exp(-0.4 * lbd) * top/bot

    val = (term1/term2) * np.sqrt(term3 + term4)

    return val