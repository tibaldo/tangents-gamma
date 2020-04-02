import numpy as np
from scipy.optimize import fsolve

####
# NB: d0 is distance projected on Galactic plane

##### to change between different coordinate systems: polar, cartesian, lon/lat/dist

def rt2xy(rad, theta):
    x = -rad * np.sin(theta)
    y = rad * np.cos(theta)
    return x, y

def rt2lond0(rad,theta,R0):
    x, y = rt2xy(rad, theta)
    lon = (np.abs(y) < R0 + (np.abs(y) > R0) * (y < 0)) * np.arctan(x / (R0 - y)) + (np.abs(y) > R0) * (
        y > 0) * (np.sign(x) * np.pi - np.arctan(x / (y - R0)))
    lon = np.rad2deg(lon)
    dist = np.sqrt(np.power(x,2)+np.power(y-R0,2))
    return lon, dist

def lbd2xy(l,b,d,R0):
    d0 = d*np.cos(np.deg2rad(b))
    r = np.sqrt(R0**2+d0**2-2*R0*d0*np.cos(np.deg2rad(l)))
    x = d0*np.sin(np.deg2rad(l))
    y = np.sqrt(r**2-x**2)
    # trick to multiply by +- 1 based on logical condition
    # that works for both numbers and numpy arrays
    b = d0 * np.cos(np.deg2rad(l)) <= R0
    i = b.astype(int)
    y *= 2 * i - 1
    #
    return x, y

#####

##### from distance to velocity and viceversa

def rt2lonvlsr(rad,theta,R0,V0,rotcurve):
    x, y = rt2xy(rad, theta)
    vel = rotcurve(rad)
    lon = (np.abs(y) < R0 + (np.abs(y) > R0) * (y < 0)) * np.arctan(x / (R0 - y)) + (np.abs(y) > R0) * (
        y > 0) * (np.sign(x) * np.pi - np.arctan(x / (y - R0)))
    vlsr = R0 * np.sin(lon) * (vel / rad - V0 / R0)
    lon = np.rad2deg(lon)
    return lon, vlsr

def lbd2vlsr(l,b,d,R0,V0,rotcurve):
    x, y = lbd2xy(l,b,d,R0)
    rad = np.sqrt(x**2+y**2)
    vel = rotcurve(rad)
    vlsr = R0 * np.sin(np.deg2rad(l)) * np.cos(np.deg2rad(b)) * (vel / rad - V0 / R0)
    return vlsr

def lbvlsr2r(l,b,vlsr,R0,V0,rotcurve):
    omega = V0 + vlsr / (np.sin(np.deg2rad(l)) * np.cos(np.deg2rad(b)))
    omega /= R0
    # helper function to find zeros of
    def F(x):
        return rotcurve(x)/x - omega
    # find zeros of F, initial guess 5 kpc
    rad = fsolve(F, 5.)
    return rad[0]

def lbvlsr2rd0(l,b,vlsr,R0,V0,rotcurve):
    # galactocentric radius
    rad = lbvlsr2r(l,b,vlsr,R0,V0,rotcurve)
    # two roots of second order-eq for d0
    a = R0 * np.cos(np.deg2rad(l))
    b = np.sqrt(np.power(rad,2) - np.power(R0 * np.sin(np.deg2rad(l)),2))
    d0_1 = a + b
    d0_2 = a - b
    d0 = np.array([d0_1,d0_2])
    # set to nan
    d0[d0 < 0.] = np.nan

    return rad, d0

#####

##### spiral arms utils
def arm_xy(s, arm_polar):
    rr, tt = arm_polar(s)
    x, y = rt2xy(rr, tt)
    return x, y

def arm_lv(s, R0, V0, arm_polar, rotcurve):
    rr, tt = arm_polar(s)
    lon,vlsr = rt2lonvlsr(rr,tt, R0, V0, rotcurve)

    return lon, vlsr

def arm_ld0(s, R0, arm_polar):
    rr, tt = arm_polar(s)
    lon,dist = rt2lond0(rr,tt,R0)

    return lon, dist