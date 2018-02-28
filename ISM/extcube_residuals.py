import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from iminuit import Minuit
import time
import pdb


class linmap_chi2:
    def __init__(self, map1, maps, maskmap):
        self.map1 = map1
        self.maps = maps
        self.maskmap = maskmap

    def model(self, pars):
        return np.sum(pars[:-1] * np.moveaxis(self.maps, 0, -1), axis=2) + pars[-1]

    def residuals(self, pars):
        return self.map1 - self.model(pars)

    def __call__(self, *pars):
        pars = np.array(pars)
        model = self.model(pars)
        residuals = self.map1 - model
        residuals *= self.maskmap
        chi2 = np.sum(np.power(residuals / np.average(self.map1), 2))
        return chi2


class extcube_residuals:
    def __init__(self, incube, inmaps, dcuts, scale=1.):
        """
        Constructor for class to create extinction residuals from extinction cube and set of
        gas maps for given distance range.
        :param incube: `string`
        FITS file with extinction cube in first HDU (from Marshall+ 2006)
        :param inmaps: `list`
        FITS file with WCS map in first HDU (gas maps)
        :param dcuts: `list`
        list of distance ranges corresponding to inmaps in kpc
        :param scale: `float`
        scaling to apply to the extinction map (so that fitting coeff are O(1))
        """

        # read extinction cube
        self.cube = fits.open(incube)[0]

        # calculate differential extinction in each distance bin
        self.cube.data[np.isnan(self.cube.data) == True] = 0.
        for s in range(len(self.cube.data) - 1, 0, -1):
            self.cube.data[s] = self.cube.data[s] - self.cube.data[s - 1]
        self.cube.data *= scale

        # read gas maps
        self.gasmaps = []
        for filename in inmaps:
            self.gasmaps.append(fits.open(filename)[0])

        # store distance cuts
        self.dcuts = dcuts

    def extmap(self, lmin, lmax, bmin, bmax, pixsize, mask='1'):
        """
        Creates a 2D extinction map from the cube for the distance cuts specified in the class
        constructors and reprojects it on the desired WCS
        :param lmin: `float`
        minimum longitude (deg)
        :param lmax: `float`
        maximum longitude (deg)
        :param bmin: `float`
        minimum latitude (deg)
        :param bmax: `float`
        maximum latitude (deg)
        :param mask: `string`
        conditions to define region where fit is performed
        :param pixsize: `float`
        pixel size (deg)
        :param mask: `str`
        conditions to use a pixel at latitude lat and longitude lon in fit (passed to Python eval),
        default '1' to accept all pixels
        :return: outmap: `~numpy.ndarray`
        output map
        :return: outheader: `~astropy.io.fits.header.Header`
        output map header
        :return: maskmap: `~numpy.ndarray`
        mask to apply during fit
        """
        # retrieve coord info from input cube
        lrefval = self.cube.header['CRVAL1']
        brefval = self.cube.header['CRVAL2']
        drefval = self.cube.header['CRVAL3']
        lrefpix = self.cube.header['CRPIX1']
        brefpix = self.cube.header['CRPIX2']
        drefpix = self.cube.header['CRPIX3']
        ldel = self.cube.header['CDELT1']
        bdel = self.cube.header['CDELT2']
        ddel = self.cube.header['CDELT3']
        ndist = self.cube.header['NAXIS3']

        # create 2D map on same spatial grid as input cube
        lbins = int((lmax - lmin) / abs(ldel)) + 1
        bbins = int((bmax - bmin) / abs(bdel)) + 1
        ldir = ldel / abs(ldel)
        bdir = bdel / abs(bdel)

        map = np.zeros([bbins, lbins])
        maskmap = np.ones([bbins, lbins])
        lpixmax = round(lrefpix + (1. / ldel) * (lmax - lrefval))
        bpixmin = round(brefpix + (1. / bdel) * (bmin - brefval))
        for ll in range(lbins):
            for bb in range(bbins):
                lpix = int(lpixmax - ll * ldir)
                bpix = int(bpixmin + bb * bdir)
                lon = lrefval + ldel * (lpix - lrefpix)
                lat = brefval + bdel * (bpix - brefpix)
                for s in range(ndist):
                    dist = drefval + ddel * (s - drefpix)
                    for drange in self.dcuts:
                        if dist >= drange[0] and dist < drange[1]:
                            map[bb, ll] += self.cube.data[s, bpix, lpix]
                if eval(mask):
                    pass
                else:
                    maskmap[bb, ll] = 0

        # create WCS object for map
        inwcs = WCS(naxis=2)  # wcs class

        # redefine bmin to avoid shifts
        bmin = brefval + bdel * (bpixmin - brefpix)
        inwcs.wcs.crpix = [int(1 + lbins / 2) + 0.5, int(1. - bmin / pixsize) + 0.5]
        inwcs.wcs.cdelt = [-abs(ldel), abs(bdel)]
        # redefine lmax and lmin to avoid shifts
        lpixmin = int(lpixmax - (lbins - 1) * ldir)
        lmin = lrefval + ldel * (lpixmin - lrefpix)
        lmax = lrefval + ldel * (lpixmax - lrefpix)
        inwcs.wcs.crval = [(lmax + lmin) / 2, 0.]
        inwcs.wcs.ctype = ['GLON-CAR', 'GLAT-CAR']
        inwcs.wcs.cunit = ['deg', 'deg']

        # create output WCS
        outwcs = WCS(naxis=2)  # wcs class
        npix = (np.array([lmax - lmin, bmax - bmin]) / pixsize).astype(int)
        outwcs.wcs.crpix = [int(1 + npix[0] / 2) + 0.5, int(1. - bmin / pixsize) + 0.5]
        outwcs.wcs.cdelt = [-pixsize, pixsize]
        outwcs.wcs.crval = [(lmax + lmin) / 2, 0.]
        outwcs.wcs.ctype = ['GLON-CAR', 'GLAT-CAR']
        outwcs.wcs.cunit = ['deg', 'deg']

        # create output header
        outheader = outwcs.to_header()
        outheader['NAXIS'] = 2
        outheader['NAXIS1'] = int(lbins * pixsize / abs(ldel)) + 1
        outheader['NAXIS2'] = int(bbins * pixsize / abs(bdel)) + 1

        # reproject output map on output wcs
        outmap, footprint = reproject_interp((map, inwcs), outheader)
        maskmap,footprint = reproject_interp((maskmap, inwcs), outheader)
        maskmap[maskmap < 1.] = 0

        return outmap, outheader, maskmap

    def reproject_gasmaps(self, outheader):
        """
        Reproject gas maps onto desired output WCS
        :param outheader: `~astropy.io.fits.header.Header`
        output map header
        :return: gasmaps: `~numpy.ndarray`
        input gas maps reprojected onto the output WCS as 3D array (#, lon, lat)
        """
        gasmaps = np.zeros([len(self.gasmaps), outheader['NAXIS2'], outheader['NAXIS1']])
        for s, inmap in enumerate(self.gasmaps):
            repromap, footrpint = reproject_interp(inmap, outheader)
            gasmaps[s] = repromap

        return gasmaps

    def fit(self, extmap, gasmaps, maskmap, outfilename='fit', outdir='./'):

        chi2 = linmap_chi2(extmap, gasmaps, maskmap)

        # define params tuple, initial values, limits, etc
        ptup = ()
        kwdarg = {}
        # map coefficients
        for n in range(len(gasmaps)):
            ptup = ptup + ('A_' + str(n),)
            kwdarg['A_' + str(n)] = 1.
            kwdarg['error_A_' + str(n)] = 0.01
            kwdarg['limit_A_' + str(n)] = (0., 1.e4)
        # constant
        ptup = ptup + ('C',)
        kwdarg['C'] = 0.
        kwdarg['error_C'] = 0.01
        kwdarg['limit_C'] = (-1.e4, 1.e4)

        # fitting
        m = Minuit(chi2, forced_parameters=ptup, errordef=1, **kwdarg)
        fitres = m.migrad()[0]

        # save results
        saveout = sys.stdout
        file = open(outdir + outfilename + '.log', 'w')
        sys.stdout = file
        print('parameters')
        for n in range(len(gasmaps)):
            print(m.values['A_' + str(n)], m.errors['A_' + str(n)])
        print(m.values['C'], m.errors['C'])
        print('FCN', m.fval, 'dof', extmap.size - len(m.args))
        print('Minuit output')
        print(fitres)
        sys.stdout = saveout
        file.close()

        # calculate residuals
        residuals = chi2.residuals(np.array(m.args))

        return fitres, residuals

    def make(self, lmin, lmax, bmin, bmax, pixsize, outfilename, outdir='./',
             mask='1', name='L. Tibaldo', email='luigi.tibaldo@irap.omp.edu'):
        """
        Make residual maps over a sky region
        :param lmin: `float`
        minimum longitude (deg)
        :param lmax: `float`
        maximum longitude (deg)
        :param bmin: `float`
        minimum latitude (deg)
        :param bmax: `float`
        maximum latitude (deg)
        :param pixsize: `float`
        pixel size (deg)
        :param outfilename: `str`
        root for the output file names
        :param outdir: `str`
        output directory
        :param mask: `str`
        conditions to use a pixel at latitude lat and longitude lon in fit (passed to Python eval),
        default '1' to accept all pixels
        :param name:
        :param email:
        :return:
        """

        # build 2D map from extinction cube in relevant distance range for required grid
        extmap, outheader, maskmap = self.extmap(lmin, lmax, bmin, bmax, pixsize, mask)

        # reproject gas maps onto output map footprint
        gasmaps = self.reproject_gasmaps(outheader)

        # set NaNs to zero
        extmap = np.nan_to_num(extmap)
        gasmaps = np.nan_to_num(gasmaps)

        # model fitting
        fitres, residuals = self.fit(extmap, gasmaps, maskmap, outfilename, outdir)

        # save residual map
        hdu = fits.PrimaryHDU(header=outheader, data=residuals)
        # add history cards
        hdu.header.add_history('map generated by {}, {}'.format(name, email))
        hdu.header.add_history('on ' + time.ctime() + ' ' + time.tzname[1])
        hdu.writeto(outdir + outfilename + '.fits')
