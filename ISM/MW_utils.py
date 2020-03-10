import numpy as np

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
    return x, y

#####

##### from distance to velocity

def rt2lonvel(rad,theta,R0,V0,rotcurve):
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
    vlsr = R0 * np.sin(np.deg2rad(l)) * (vel / rad - V0 / R0)
    return vlsr

#####

##### spiral arms utils
def arm_xy(s, arm_polar):
    rr, tt = arm_polar(s)
    x, y = rt2xy(rr, tt)
    return x, y

def arm_lv(s, R0, V0, arm_polar, rotcurve):
    rr, tt = arm_polar(s)
    lon,vlsr = rt2lonvel(rr,tt, R0, V0, rotcurve)

    return lon, vlsr

def arm_ld0(s, R0, arm_polar):
    rr, tt = arm_polar(s)
    lon,dist = rt2lond0(rr,tt,R0)

    return lon, dist