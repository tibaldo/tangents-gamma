import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_from_healpix
from iminuit import Minuit
import time


class linmap_chi2:
    def __init__(self, map1, maps, maskmap, errormap):
        self.map1 = map1
        self.maps = maps
        self.maskmap = maskmap
        self.errormap = errormap

    def model(self, pars):
        return np.sum(pars[:-1] * np.moveaxis(self.maps, 0, -1), axis=2) + pars[-1]

    def residuals(self, pars):
        return self.map1 - self.model(pars)

    def __call__(self, *pars):
        pars = np.array(pars)
        model = self.model(pars)
        residuals = self.map1 - model
        residuals /= self.errormap
        residuals *= self.maskmap
        chi2 = np.sum(np.power(residuals, 2))
        return chi2


class dustmap_residuals:
    def __init__(self, dustmap, colname, inmaps, scale=1., errorname='None'):
        """
        Constructor for class to create extinction residuals from extinction cube and set of
        gas maps for given distance range.
        :param dustmap: `string`
        FITS file with dust map
        :param colname: `string`
        name of column containing dust map
        :param inmaps: `list`
        FITS file with WCS map in first HDU (gas maps)
        :param scale: `float`
        scaling to apply to the extinction map (so that fitting coeff are O(1))
        :param scale: `string`
        name of column containing error map, 'None' for no error
        """

        # read in dust map
        self.dustmap = fits.open(dustmap)[1]
        self.colname = colname
        self.scale = scale
        self.errorname = errorname

        # read gas maps
        self.gasmaps = []
        self.nreg = len(inmaps)
        self.region = []
        for s, region in enumerate(inmaps):
            for filename in region:
                self.region.append(s)
                self.gasmaps.append(fits.open(filename)[0])

    def reproject_dustmap(self, outheader):
        """
        Reprojects dust map (and errors) on the desired WCS
        :param outheader: `~astropy.io.fits.header.Header`
        output map header
        :return: outmap: `~numpy.ndarray`
        output map
        :return: errormap: `~numpy.ndarray`
        erromap
        """

        # load properties of healpix grid
        coordsys = self.dustmap.header['COORDSYS']
        if coordsys == 'C':
            coordsys = 'ICRS'
        elif coordsys == 'E':
            coordsys = 'Ecliptic'
        elif coordsys == 'G':
            coordsys = 'Galactic'
        else:
            print('coordinate system of input dust map unknown:', coordsys)

        nested = self.dustmap.header['ORDERING']
        if nested == 'NESTED':
            nested = True
        elif nested == 'RING':
            nested = False
        else:
            print('ordering of input dust map unknown:', nested)

        # dust map
        outmap, footprint = reproject_from_healpix((self.dustmap.data[self.colname], coordsys),
                                                   outheader, nested=nested)

        # error map
        if self.errorname == 'None':
            errormap = np.ones(np.shape(outmap))
        else:
            errormap, footprint = reproject_from_healpix(
                (self.dustmap.data[self.errorname], coordsys),
                outheader, nested=nested)

        outmap *= self.scale
        errormap *= self.scale

        return outmap, errormap

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

    def fit(self, extmap, gasmaps, maskmap, errormap, outfilename='fit', outdir='./',
            split=True):

        chi2 = linmap_chi2(extmap, gasmaps, maskmap, errormap)

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

        # calculate weights of each region
        parvals = np.array(m.args)
        parvals[-1] = 0  # remove constant
        total_model = chi2.model(parvals)
        weights = []
        for s in range(self.nreg):
            parvals = np.array(m.args)
            for k in range(len(self.region)):
                if self.region[k] == s:
                    pass
                else:
                    parvals[k] = 0.
            model_reg = chi2.model(parvals)
            weight = model_reg / total_model
            weight[total_model < 0.1] = 0.
            weights.append(weight)

        return fitres, residuals, weights

    def make(self, lmin, lmax, bmin, bmax, pixsize, outfilename, names, outdir='./',
             mask='None', name='L. Tibaldo', email='luigi.tibaldo@irap.omp.eu'):
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
        default 'None' to accept all pixels
        :param name:
        :param email:
        :return:
        """

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
        outheader['NAXIS1'] = int((lmax - lmin) / pixsize) + 1
        outheader['NAXIS2'] = int((bmax - bmin) / pixsize) + 1

        # reproject input map onto required grid
        dustmap, errormap = self.reproject_dustmap(outheader)

        # reproject gas maps onto output map footprint
        gasmaps = self.reproject_gasmaps(outheader)

        # create mask map
        maskmap = np.ones(np.shape(dustmap))
        if mask == 'None':
            pass
        else:
            for ll in range(outheader['NAXIS1']):
                for bb in range(outheader['NAXIS2']):
                    lon = outwcs.wcs.crval[0] + outwcs.wcs.cdelt[0] * (
                    ll - outwcs.wcs.crpix[0])
                    lat = outwcs.wcs.crval[1] + outwcs.wcs.cdelt[1] * (
                    bb - outwcs.wcs.crpix[1])
                    if eval(mask):
                        pass
                    else:
                        maskmap[bb, ll] = 0

        # set NaNs to zero
        dustmap = np.nan_to_num(dustmap)
        gasmaps = np.nan_to_num(gasmaps)
        errormap = np.nan_to_num(errormap)
        # set to 1 error if == 0
        errormap[errormap == 0.] = 1.

        print('Finished reprojecting maps, starting fit')

        # model fitting
        fitres, residuals, weights = self.fit(dustmap, gasmaps, maskmap, errormap, outfilename,
                                              outdir, split=True)

        # save total residual map
        hdu = fits.PrimaryHDU(header=outheader, data=residuals)
        # add history cards
        hdu.header.add_history('map generated by {}, {}'.format(name, email))
        hdu.header.add_history('on ' + time.ctime() + ' ' + time.tzname[1])
        hdu.writeto(outdir + outfilename + '.fits')

        # save splitted residuals
        for s in range(self.nreg):
            split_residuals = residuals * weights[s]
            split_residuals[split_residuals < 0] = 0.
            hdu = fits.PrimaryHDU(header=outheader, data=split_residuals)
            # add history cards
            hdu.header.add_history('map generated by {}, {}'.format(name, email))
            hdu.header.add_history('on ' + time.ctime() + ' ' + time.tzname[1])
            hdu.writeto(outdir + outfilename + '_{}.fits'.format(names[s]))
