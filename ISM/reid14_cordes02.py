import numpy as np

######################################################
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


def V(R):
    x = R / (a2 * R0)
    val = beta * 1.97 * x ** 1.22 / (x ** 2 + 0.78 ** 2) ** 1.43
    val += (1 - beta) * (1 + a3 ** 2) * x ** 2 / (x ** 2 + a3 ** 2)
    val *= a1 ** 2
    val = np.sqrt(val)
    return val


########################################################

########################################################
##spiral arm model, model from Wainscoat 1992
##with rescuplting by Cordes2002
##translated from NE2001 fortran routine
Narm = 5
arms = [[4.25, 3.48, 0., 7.],
        [4.25, 3.48, 3.141, 6.],
        [4.89, 4.90, 2.525, 6.],
        [4.89, 3.76, 4.24, 8.],
        [4.57, 8.10, 5.847, 0.55]]


def rt2xy(rad, theta):
    x = -rad * np.sin(theta)
    y = rad * np.cos(theta)
    return x, y

def rt2lonvel(rad,theta):
    x, y = rt2xy(rad, theta)
    vel = V(rad)
    lon = (np.abs(y) < R0 + (np.abs(y) > R0) * (y < 0)) * np.arctan(x / (R0 - y)) + (np.abs(y) > R0) * (
        y > 0) * (np.sign(x) * np.pi - np.arctan(x / (y - R0)))
    vlsr = R0 * np.sin(lon) * (vel / rad - V0 / R0)
    lon = np.rad2deg(lon)
    return lon, vlsr

def rt2lond0(rad,theta):
    x, y = rt2xy(rad, theta)
    lon = (np.abs(y) < R0 + (np.abs(y) > R0) * (y < 0)) * np.arctan(x / (R0 - y)) + (np.abs(y) > R0) * (
        y > 0) * (np.sign(x) * np.pi - np.arctan(x / (y - R0)))
    lon = np.rad2deg(lon)
    dist = np.sqrt(np.power(x,2)+np.power(y-R0,2))
    return lon, dist

def lbd2vlsr(l,b,d):
    x, y = lbd2xy(l,b,d)
    rad = np.sqrt(x**2+y**2)
    vel = V(rad)
    vlsr = R0 * np.sin(np.deg2rad(l)) * (vel / rad - V0 / R0)
    return vlsr

def lbd2xy(l,b,d):
    d0 = d*np.cos(np.deg2rad(b))
    r = np.sqrt(R0**2+d0**2-2*R0*d0*np.cos(np.deg2rad(l)))
    x = d0*np.sin(np.deg2rad(l))
    y = np.sqrt(r**2-x**2)
    return x, y

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
    rr *= R0/8.5  # Wainscoat 1992 uses Rsun=8.5, why don't I need to correct if I change this value?
    return rr, tt

def arm_xy(s):
    rr, tt = arm_polar(s)
    x, y = rt2xy(rr, tt)
    return x, y

def arm_lv(s):
    rr, tt = arm_polar(s)
    lon,vlsr = rt2lonvel(rr,tt)

    return lon, vlsr

def arm_ld0(s):
    rr, tt = arm_polar(s)
    lon,dist = rt2lond0(rr,tt)

    return lon, dist
