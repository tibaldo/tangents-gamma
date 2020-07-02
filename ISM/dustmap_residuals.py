import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_from_healpix
from iminuit import Minuit
import scipy.ndimage as ni
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time
import pdb


def filter_map(inmap, vmin=-200, vmax=200, bin=5, smooth_radius=4, threshold=2.,
               figname='histo.png'):
    # create a smoothed map
    smoothmap = ni.gaussian_filter(inmap.astype("d"), smooth_radius)

    # create residual histogram
    fig = plt.figure('Residuals')
    ax = plt.subplot(111)
    bindef = np.arange(vmin, vmax, bin)
    hist_vals, binEdges, patches = plt.hist(smoothmap.flatten(), bindef, histtype='stepfilled')
    #plt.setp(patches, 'facecolor', 'white', 'alpha', 0.75)
    binCenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    ax.set_xlabel('residuals')
    ax.set_ylabel('pixels')
    ax.set_xlim(vmin, vmax)
    ax.set_yscale('log')
    ax.set_ylim(1, 1.e5)

    # fit residual histogram with Gaussian
    fitfunc = lambda p, x: p[0] * np.exp(-np.power((x - p[1]) / p[2], 2) / 2)
    errfunc = lambda p, x, y: (fitfunc(p, x) - y)  # /np.sqrt(y)
    # make educated guesses on the true param values
    data_std = np.std(smoothmap.flatten())
    p0 = [np.max(hist_vals), 0., data_std]
    fitrange = np.where(np.abs(binCenters) < 3 * data_std)
    p1, flag = opt.leastsq(errfunc, p0, args=(binCenters[fitrange], hist_vals[fitrange]))
    pltpts = np.linspace(vmin, vmax, 1000)
    ax.plot(pltpts, fitfunc(p1, pltpts), color='r')

    # filter input map and save
    cutval = threshold * p1[2]
    newmap = inmap
    newmap[smoothmap < cutval] = 1.e-6
    newmap[newmap < 1.e-6] = 1.e-6

    plt.savefig(figname)
    plt.close(fig.number)
    del ax

    return newmap


class linmap_chi2:
    def __init__(self, map1, maps, maskmap, errormap,
                 error_mode,error_frac):
        self.map1 = map1
        self.maps = maps
        self.maskmap = maskmap
        self.errormap = errormap
        self.error_mode = error_mode
        self.error_frac = error_frac

    def model(self, pars):
        return np.sum(pars[:-1] * np.moveaxis(self.maps, 0, -1), axis=2) + pars[-1]

    def residuals(self, pars):
        return self.map1 - self.model(pars)

    def __call__(self, *pars):
        pars = np.array(pars)
        model = self.model(pars)
        residuals = self.map1 - model
        if self.error_mode == 'ERROR':
            residuals /= self.errormap
        if self.error_mode == 'DATA':
            residuals /= self.error_frac * self.map1
            residuals[self.map1==0] = 0.
        if self.error_mode == 'MODEL':
            residuals /= self.error_frac * model
            residuals[model==0] = 0.
        residuals *= self.maskmap
        chi2 = np.sum(np.power(residuals, 2))
        return chi2


class dustmap_residuals:
    def __init__(self, dustmap, mapname, inmaps, scale=1., errorname='None',hpx=True):
        """
        Constructor for class to create extinction residuals from dust map and set of
        gas maps for given distance range.
        :param dustmap: `string`
        FITS file with dust map
        :param mapname: `string`
        name of column (HPX) or HDU (WCS) containing dust map
        :param inmaps: `list`
        FITS file with WCS map in first HDU (gas maps)
        :param scale: `float`
        scaling to apply to the extinction map (so that fitting coeff are O(1))
        :param scale: `string`
        name of column containing error map, 'None' for no error
        """

        # read in dust map
        self.dustmap = fits.open(dustmap)
        self.mapname = mapname
        self.scale = scale
        self.errorname = errorname
        self.hpx = hpx

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

        if self.hpx:
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
            if self.dustmap.header['TFORM1'] == '1024E':
                outmap, footprint = reproject_from_healpix(self.dustmap[1],outheader)
            else:
                outmap, footprint = reproject_from_healpix((self.dustmap[1].data[self.mapname], coordsys),
                                                           outheader, nested=nested)

            # error map
            if self.errorname == 'None':
                errormap = np.ones(np.shape(outmap))
            else:
                errormap, footprint = reproject_from_healpix(
                    (self.dustmap[1].data[self.errorname], coordsys),
                    outheader, nested=nested)
        else:
            outmap, footprint = reproject_interp(self.dustmap[self.mapname], outheader)
            if self.errorname == 'None':
                errormap = np.ones(np.shape(outmap))
            else:
                errormap, footprint = reproject_interp(self.dustmap[self.errorname], outheader)

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

    def fit(self, extmap, gasmaps, maskmap, errormap,
            error_mode,error_frac,
            outfilename='fit', outdir='./',
            split=False):

        chi2 = linmap_chi2(extmap, gasmaps, maskmap, errormap,
                           error_mode,error_frac)

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
        outvals = np.array([])
        outerrs = np.array([])
        saveout = sys.stdout
        file = open(outdir + outfilename + '.log', 'w')
        sys.stdout = file
        print('parameters')
        for n in range(len(gasmaps)):
            print(m.values['A_' + str(n)], m.errors['A_' + str(n)])
            outvals = np.append(outvals,m.values['A_' + str(n)])
            outerrs = np.append(outerrs,m.errors['A_' + str(n)])
        print(m.values['C'], m.errors['C'])
        outvals = np.append(outvals, m.values['C'])
        outerrs = np.append(outerrs, m.errors['C'])
        print('FCN', m.fval, 'dof', np.sum(maskmap) - len(m.args))
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
            if split:
                parvals = np.array(m.args)
                for k in range(len(self.region)):
                    if self.region[k] == s:
                        pass
                    else:
                        parvals[k] = 0.
                model_reg = chi2.model(parvals)
                weight = model_reg / total_model
                weight[total_model < 0.1] = 0.
            else:
                weight = np.zeros(np.shape(total_model))
            weights.append(weight)

        return residuals, weights, m.fval, outvals, outerrs


    def make(self, lmin, lmax, bmin, bmax, pixsize, outfilename, names,
             error_mode,error_frac=1.0,
             outdir='./', split=False,
             max_iter = 1, smooth_radius = 4., threshold=2, nsig = 1,
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
        residuals, weights, chi2, vals, errs = self.fit(dustmap, gasmaps, maskmap, errormap,
                                                        error_mode, error_frac,
                                                        outfilename, outdir, split=split)

        dnmmap = filter_map(residuals,
                            smooth_radius=smooth_radius, threshold=threshold,
                            figname=outdir + 'histo_0.png')

        if max_iter == 1:
            pass
        else:
            history = {}
            history[0] = {}
            history[0]['values'] = np.append(np.append(vals[:-1],np.nan),vals[-1])
            history[0]['errors'] = np.append(np.append(errs[:-1],np.nan),errs[-1])
            history[0]['chi2'] = chi2
            chi2diff = 10000
            pardiff = 10000 * np.ones(len(gasmaps) + 2)
            gasmaps = np.concatenate((gasmaps,[dnmmap]),axis=0)
            niter = 1
            while niter < max_iter and (chi2diff > 1 or np.any(pardiff > nsig)):
                old_chi2 = chi2
                old_vals = vals
                gasmaps[-1] = dnmmap
                residuals, weights, chi2, vals, errs = self.fit(dustmap, gasmaps, maskmap,
                                                                errormap,
                                                                error_mode, error_frac,
                                                                outfilename+'_{}'.format(niter), outdir,
                                                                split=False)
                history[niter] = {}
                history[niter]['values'] = vals
                history[niter]['errors'] = errs
                history[niter]['chi2'] = chi2
                if niter > 1:
                    chi2diff = np.abs(chi2-old_chi2)
                    pardiff = np.abs(vals-old_vals)/errs
                dnmmap = filter_map(residuals + vals[-2] * gasmaps[-1],
                                    smooth_radius=smooth_radius, threshold=threshold,
                                    figname=outdir + 'histo_{}.png'.format(niter))
                niter += 1


        # save DNM map
        hdu = fits.PrimaryHDU(header=outheader, data=dnmmap)
        # add history cards
        hdu.header.add_history('map generated by {}, {}'.format(name, email))
        hdu.header.add_history('on ' + time.ctime() + ' ' + time.tzname[1])
        hdu.writeto(outdir + outfilename + '.fits')

        # save residual map
        hdu = fits.PrimaryHDU(header=outheader, data=residuals)
        # add history cards
        hdu.header.add_history('map generated by {}, {}'.format(name, email))
        hdu.header.add_history('on ' + time.ctime() + ' ' + time.tzname[1])
        hdu.writeto(outdir + outfilename + '_resid.fits')

        # save splitted DNM map
        if split:
            for s in range(self.nreg):
                split_residuals = dnmmap * weights[s]
                split_residuals[split_residuals < 0] = 0.
                hdu = fits.PrimaryHDU(header=outheader, data=split_residuals)
                # add history cards
                hdu.header.add_history('map generated by {}, {}'.format(name, email))
                hdu.header.add_history('on ' + time.ctime() + ' ' + time.tzname[1])
                hdu.writeto(outdir + outfilename + '_{}.fits'.format(names[s]))

        # save fit history
        if max_iter > 1:
            np.save(outdir + 'history.npy', history)
            # chi2
            fig = plt.figure()
            ax = plt.subplot()
            ax.set_xlabel('iteration step')
            ax.set_ylabel(r'$\chi^2$')
            vals = []
            for s in range(niter):
                vals.append(history[s]['chi2'])
            ax.plot(vals,color='k',marker='o',linewidth=0.)
            fig.savefig(outdir + 'chi2.png')
            # parameters
            for ipar in range(len(history[1]['values'])):
                fig = plt.figure()
                ax = plt.subplot()
                ax.set_xlabel('iteration step')
                ax.set_ylabel('parameter {}'.format(ipar))
                vals = []
                errs = []
                for s in range(niter):
                    vals.append(history[s]['values'][ipar])
                    errs.append(history[s]['errors'][ipar])
                ax.errorbar(np.arange(niter),vals,yerr=errs,fmt='ko')
                fig.savefig(outdir + 'parameter_{}.png'.format(ipar))


