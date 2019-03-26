import numpy as np
import pdb
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from reid14_cordes02 import lbd2vlsr
from reproject import reproject_exact, reproject_interp


def merge_maps(maps, outmap, target_res, lmin, lmax, bmin, bmax,vmin,vmax,
               dcuts,ireg):
    # create output hdu
    # determine map properties
    nx = int((lmax - lmin) / target_res)
    dx = (lmax - lmin) / nx
    ny = int((bmax - bmin) / target_res)
    dy = (bmax - bmin) / ny
    crpix_y = -bmin / dy + 1
    # wcs
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [1., crpix_y]
    w.wcs.cdelt = np.array([-dx, dy])
    w.wcs.crval = [lmax, 0.]
    w.wcs.ctype = ["GLON-CAR", "GLAT-CAR"]
    # header
    header = w.to_header()
    # empty map
    map_data = np.zeros([ny, nx])
    # hdu
    hdu = fits.PrimaryHDU(map_data, header=header)

    # reproject input maps onto output map
    reprojected_maps = []
    for map in maps:
        # reproject
        array, footprint = reproject_exact(fits.open(map)[0], hdu.header)
        # mask pixels affected by anomalies
        array = mask_map(array,hdu.header,dcuts,ireg,vmin,vmax)
        reprojected_maps.append(array)

    # fill output map with input giving precedence to maps at the end of the list
    for map in reprojected_maps:
        hdu.data[map > 0] = map[map > 0]

    # write new file
    hdu.writeto(outmap)


def mask_map(data, header, anomalies, dcuts, ireg, vmin, vmax):
    # find min/max distance
    dmin = False
    dmax = False
    if ireg - 1 >= 0:
        try:
            dmin = dcuts[ireg - 1]
        except:
            pass
    else:
        pass
    try:
        dmax = dcuts[ireg]
    except:
        pass

    # read anomalies
    anomalies = np.genfromtxt(anomalies, delimiter=',')
    # fix for now, get rid of it
    anomalies[:, 2] /= 1.e3

    # find min/max velocity
    if dmin:
        vmin = lbd2vlsr(anomalies[:, 0], anomalies[:, 1], [dmin] * len(anomalies))
    else:
        vmin = np.array([vmin] * len(anomalies))
    if dmax:
        vmax = lbd2vlsr(anomalies[:, 0], anomalies[:, 1], [dmax] * len(anomalies))
    else:
        vmax = np.array([vmax] * len(anomalies))

    # filter anomalies that are irrelevant
    anomalies = anomalies[(anomalies[:, 2] >= vmin) & (anomalies[:, 2] <= vmax)]

    # reduce number of anomaly positions by eliminating duplicates
    anomalies = anomalies[:, :2]
    anomalies = np.unique(anomalies, axis=0)

    # convert anomalies in sky coord object
    an_c = SkyCoord(anomalies[:, 0], anomalies[:, 1], frame="galactic", unit="deg")

    # build mesh grid of input map pixel coordinates
    w = wcs.WCS(header)
    x = np.arange(header['NAXIS1'])
    y = np.arange(header['NAXIS2'])
    X, Y = np.meshgrid(x, y)
    lon, lat = w.wcs_pix2world(X, Y, 0)

    # convert mesh grid to sky coord
    map_c = SkyCoord(lon, lat, frame="galactic", unit="deg")

    dist = [map_c.separation(anomaly).deg for anomaly in an_c]
    dist = np.array(dist)
    dist = dist.min(axis=0)

    mask = np.ones(np.shape(data))
    min_sep = np.maximum(np.abs(header['CDELT1']),np.abs(header['CDELT2']))
    min_sep *= np.sqrt(2)/2
    mask[dist < min_sep] = 0.

    data *= mask

    return data
