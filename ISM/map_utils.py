import numpy as np
import pdb
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from reid19_rotation import rotcurve, R0, V0
from MW_utils import lbd2vlsr
from reproject import reproject_exact, reproject_interp
import cv2
import pdb


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
        array, footprint = reproject_interp(fits.open(map)[0], hdu.header)
        hdu.data[footprint == 1] = array[footprint == 1]

    # write new file
    hdu.writeto(outmap)


def create_anomaly_mask(header, anomalies, vmin, vmax, ireg, mask_radius = 'NONE', border = 0,
                        vcuts=False, dcuts=False, cutfile = False,
                        save = False, outfilename='mask.fits'):

    # mask will be one in pixels to mask, 0 otherwise
    # default set to None
    mask = 'NONE'

    # read anomalies
    anomalies = np.genfromtxt(anomalies, delimiter=',')

    if len(anomalies) > 0 :
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
                vmin = lbd2vlsr(anomalies[:, 0], anomalies[:, 1], [dmin] * len(anomalies),R0,V0,rotcurve)
            if ireg < len(dcuts):
                dmax = dcuts[ireg]
                vmax = lbd2vlsr(anomalies[:, 0], anomalies[:, 1], [dmax] * len(anomalies),R0,V0,rotcurve)
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
            if ireg < len(cuts):
                cut = cuts[ireg]
                vmax = np.interp(anomalies[:, 0], cut[0], cut[1])
        else:
            # use overall velocity range
            pass

        # filter anomalies that are irrelevant (outside velocity range)
        anomalies = anomalies[(anomalies[:, 2] >= vmin) & (anomalies[:, 2] <= vmax)]

        if len(anomalies) > 0:

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

            # mask radius = pixel size if not specified otherwise
            if mask_radius == 'NONE':
                mask_radius = np.maximum(np.abs(header['CDELT1']),np.abs(header['CDELT2']))
            # add border
            mask_radius *= (1 + border)

            mask = np.zeros(np.shape(lon))
            mask[dist < mask_radius] = 1.

            if save:
                hdu = fits.PrimaryHDU(mask,header)
                hdu.header.add_comment('Anomaly mask with radius {} deg'.format(mask_radius))
                hdu.writeto(outfilename)

    return mask

def inpaint_map(infile,mask,outfilename,mask_radius=3):

    # read data
    data = fits.getdata(infile)

    # transform data in grayscale img ([0,255]) readable by opencv
    bl = np.min(data)
    scale = 255 / np.max(data - bl)
    img = np.array(scale * (data - bl), dtype=np.uint8)

    # transform mask
    mask = np.array(mask * 255, dtype=np.uint8)

    # inpaint
    outimg = cv2.inpaint(img, mask, mask_radius, cv2.INPAINT_TELEA)

    # scale back to physical values
    outimg = outimg/scale + bl

    # replace masked values with inpainted ones
    data[mask>0] = outimg[mask>0]

    # write file
    hdu = fits.PrimaryHDU(data, fits.getheader(infile))
    hdu.header.add_comment('Anomalies inpainted within {} pixels'.format(mask_radius))
    hdu.writeto(outfilename)


def rebin_map(infile,outfile, target_res):

    # read input map hdu
    hdu = fits.open(infile)[0]

    # number of current pixels
    npix_lat, npix_lon = np.shape(hdu.data)

    # rebin factor
    rebin1 = int(target_res / np.abs(hdu.header['CDELT1']))
    rebin2 = int(target_res / np.abs(hdu.header['CDELT2']))
    rebin = np.minimum(rebin1,rebin2)
    npix_lon_reb = int(npix_lon / rebin)
    npix_lat_reb = int(npix_lat / rebin)

    # create and fill outmput map
    print('rebinning to create '+outfile)
    outmap = np.zeros([npix_lat_reb, npix_lon_reb])
    # map is filled from a corner
    # will trim edges if # number of input pixels not exactly equals rebin * # output pixels
    for j in range(npix_lat_reb):
        for i in range(npix_lon_reb):
            outmap[j, i] = np.average(hdu.data[j * rebin:(j + 1) * rebin, i * rebin:(i + 1) * rebin])
    print('finished rebinning to create ' + outfile)

    # output HDU and header
    outhdu = fits.PrimaryHDU(outmap,hdu.header)
    # tweak header values to match rebinned map
    outhdu.header['CRPIX1'] = 1
    outhdu.header['CDELT1'] = hdu.header['CDELT1'] * rebin
    outhdu.header['CRVAL1'] = hdu.header['CRVAL1'] + hdu.header['CDELT1'] * 0.5 * (rebin - 1)
    oldlat0pix = - hdu.header['CRVAL2'] / hdu.header['CDELT2'] + hdu.header['CRPIX2'] - 1
    newlat0pix = 1 + int((oldlat0pix + 1) / rebin) +\
                 ((oldlat0pix + 1) % rebin)/rebin +0.5 * (1./rebin  - 1)
    outhdu.header['CRPIX2'] = newlat0pix
    outhdu.header['CDELT2'] = hdu.header['CDELT2'] * rebin
    outhdu.header['CRVAL2'] = 0.
    comment = 'Rebinned by factor {} w.r.t. to native resolution'.format(rebin)
    outhdu.header.add_comment(comment)

    # write file
    print('saving ' + outfile)
    outhdu.writeto(outfile)
    print('saved ' + outfile)







