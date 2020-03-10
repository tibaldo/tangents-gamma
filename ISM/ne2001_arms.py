import numpy as np
##spiral arm model, model from Wainscoat 1992
##with rescuplting by Cordes2002
##translated from NE2001 fortran routine

Narm = 5
arms = [[4.25, 3.48, 0., 7.],
        [4.25, 3.48, 3.141, 6.],
        [4.89, 4.90, 2.525, 6.],
        [4.89, 3.76, 4.24, 8.],
        [4.57, 8.10, 5.847, 0.55]]


def arm_polar(s):
    params = arms[s]
    rr = np.logspace(np.log10(params[1]),np.log10(30), 200)
    tt = params[0] * np.log(rr / params[1]) + params[2]
    rr = rr[tt < params[2] + params[3]]
    tt = tt[tt < params[2] + params[3]]
    if s == 1:
        rr[(tt > np.deg2rad(370)) & (tt < np.deg2rad(410))] *= 1. + 0.04 * np.cos(
            (tt[(tt > np.deg2rad(370)) & (tt < np.deg2rad(410))] - np.deg2rad(390.)) * 180. / 40.)
        rr[(tt > np.deg2rad(315)) & (tt < np.deg2rad(370))] *= 1. - 0.07 * np.cos(
            (tt[(tt > np.deg2rad(315)) & (tt < np.deg2rad(370))] - np.deg2rad(345.)) * 180. / 55.)
        rr[(tt > np.deg2rad(180)) & (tt < np.deg2rad(315))] *= 1. + 0.16 * np.cos(
            (tt[(tt > np.deg2rad(180)) & (tt < np.deg2rad(315))] - np.deg2rad(260.)) * 180. / 135.)
    elif s == 3:
        rr[(tt > np.deg2rad(290)) & (tt < np.deg2rad(395))] *= 1. -0.11 * np.cos(
            (tt[(tt > np.deg2rad(290)) & (tt < np.deg2rad(395))] - np.deg2rad(350.)) * 180. / 105.)
    return rr, tt
