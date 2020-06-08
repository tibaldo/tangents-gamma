import numpy as np
import pdb
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from reid14_cordes02 import lbd2vlsr
from reproject import reproject_exact, reproject_interp


def merge_maps(maps, outmap, target_res, lmin, lmax, bmin, bmax):
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
    # and fill output map with input giving precedence to maps at the end of the list
    for map in maps:
        # reproject
        array, footprint = reproject_exact(fits.open(map)[0], hdu.header)
        hdu.data[footprint == 1] = array[footprint == 1]

    # write new file
    hdu.writeto(outmap)


def create_anomaly_mask(header, anomalies, vmin, vmax, ireg, mask_radius,
                        vcuts=False, dcuts=False, cutfile = False,
                        save = False, outfilename='mask.fits'):

    # mask will be one in pixels to mask, 0 otherwise

    # read anomalies
    anomalies = np.genfromtxt(anomalies, delimiter=',')

    # find min/max velocity
    # start with overall boundaries
    vmin = np.array([vmin] * len(anomalies))
    vmax = np.array([vmax] * len(anomalies))
    if vcuts:
        if ireg > 0:
            vmin = np.array([vcuts[ireg][0]] * len(anomalies))
        if ireg < len(vcuts) - 1:
            vmax = np.array([vcuts[ireg][1]] * len(anomalies))
    elif dcuts:
        if ireg > 0:
            dmin = dcuts[ireg-1]
            vmin = lbd2vlsr(anomalies[:, 0], anomalies[:, 1], [dmin] * len(anomalies))
        if ireg < len(dcuts):
            dmax = dcuts[ireg]
            vmax = lbd2vlsr(anomalies[:, 0], anomalies[:, 1], [dmax] * len(anomalies))
    elif cutfile:
        cuts = np.load(cutfile)
        # make sure x-values are sorted
        for cut in cuts:
            idx = np.argsort(cut[0])
            cut[1] = cut[1][idx]
            cut[0] = cut[0][idx]
        if ireg > 0:
            cut = cuts[ireg-1]
            vmin = np.interp(anomalies[:, 0],cut[0],cut[1])
        if ireg < len(dcuts):
            cut = cuts[ireg]
            vmax = np.interp(anomalies[:, 0], cut[0], cut[1])
    else:
        # use overall velocity range
        pass

    # filter anomalies that are irrelevant (outside velocity range)
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

    mask = np.zeros(np.shape(lon))
    mask[dist < mask_radius] = 1.

    if save:
        hdu = fits.PrimaryHDU(mask,header)
        hdu.header['RECORD'] = 'Anomaly mask with radius {} deg'.format(radius)
        hdu.writeto(outfilename)

    return data
