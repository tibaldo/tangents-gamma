import matplotlib.pyplot as plt
import numpy as np
import time
import gzip
from astropy.io import fits
from epsDetectS import epsDetect
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import AxesGrid
from pseudoVoigt import pseudoVoigt  # Quentin's version
from reid19_rotation import rotcurve, R0, V0
from MW_utils import lbd2vlsr
from scipy.io import loadmat
import pdb


def add_inner_title(ax, title, loc, size=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at


class gascube:
    def __init__(self, filename, int2col=1., Ts=-10, fitres_files=None):

        hdus = fits.open(filename)

        # store header
        self.header = hdus[0].header

        # read the axis type and mapping values
        naxis = self.header.get('NAXIS')
        self.atlas = {}
        self.refpix = {}
        self.refval = {}
        self.delta = {}
        self.naxis = {}
        for i in range(naxis):
            if (self.header.get('CTYPE' + str(i + 1)) == 'GLON-CAR'):
                self.atlas['longitude'] = i + 1
                self.refpix['longitude'] = self.header.get(
                    'CRPIX' + str(i + 1)) - 1
                self.refval['longitude'] = self.header.get('CRVAL' + str(i + 1))
                self.delta['longitude'] = self.header.get('CDELT' + str(i + 1))
                self.naxis['longitude'] = self.header.get('NAXIS' + str(i + 1))
            if (self.header.get('CTYPE' + str(i + 1)) == 'GLAT-CAR'):
                self.atlas['latitude'] = i + 1
                self.refpix['latitude'] = self.header.get(
                    'CRPIX' + str(i + 1)) - 1
                self.refval['latitude'] = self.header.get('CRVAL' + str(i + 1))
                self.delta['latitude'] = self.header.get('CDELT' + str(i + 1))
                self.naxis['latitude'] = self.header.get('NAXIS' + str(i + 1))
            if (self.header.get(
                    'CTYPE' + str(i + 1)) == 'VELO-LSR' or self.header.get(
                'CTYPE' + str(i + 1)) == 'VELO-LSRK' or self.header.get(
                'CTYPE' + str(i + 1)) == 'VEL' or self.header.get(
                'CTYPE' + str(i + 1)) == 'VRAD'):
                # initialise velocity unit
                self.vscale = 1.
                self.atlas['velocity'] = i + 1
                self.refpix['velocity'] = self.header.get(
                    'CRPIX' + str(i + 1)) - 1
                self.refval['velocity'] = self.header.get('CRVAL' + str(i + 1))
                self.delta['velocity'] = self.header.get('CDELT' + str(i + 1))
                self.naxis['velocity'] = self.header.get('NAXIS' + str(i + 1))
                # check if velocity unit is stored in comment
                if 'M/S' in self.header.comments['CDELT' + str(i + 1)]:
                    self.vscale = 1.e3
                if 'KM/S' in self.header.comments['CDELT' + str(i + 1)]:
                    self.vscale = 1.
                # check if velocity unit is defined by dedicated key
                # key takes precedence over comment
                try:
                    u = self.header.get('CUNIT' + str(i + 1))
                    if u == 'M/S' or u == 'm/s' or u == 'm s-1':
                        self.vscale = 1.e3
                    else:
                        pass
                except:
                    pass

        # find the value assigned to blank pixels
        try:
            bzero = self.header.get('BZERO')
            bscale = self.header.get('BSCALE')
            blank = self.header.get('BLANK')
            blankvalue = bzero + bscale * blank
        except:
            blankvalue = -10000

        # open data and set to 0 blank pixels
        self.data = hdus[0].data
        if naxis == 3:
            self.data = hdus[0].data
        elif naxis == 4:
            self.data = hdus[0].data[0, :, :, :]
        else:
            print("ERROR, anomalous number of axes in FITS file", filename)
        self.data = np.nan_to_num(self.data)
        self.data[self.data <= (blankvalue + 0.1)] = 0.

        self.int2col = int2col
        self.Ts = Ts  # default=-10 is optically thin approx

        # read fit results file if available, and set bool to True
        self.fitres = {}
        self.fitres['available'] = False
        if not fitres_files == None:
            try:
                fitres = np.load(fitres_files[0])
                fitdiag = np.load(fitres_files[1])
                self.fitres['available'] = True
            except:
                try:
                    fitres = loadmat(fitres_files[0])
                    fitdiag = loadmat(fitres_files[1])
                    self.fitres['available'] = True
                except:
                    pass
        if self.fitres['available']:
            self.fitres['vlin'] = fitres['vlin']
            self.fitres['hfit'] = fitres['hfit']
            self.fitres['vfit'] = fitres['vfit']
            self.fitres['svfit'] = fitres['svfit']
            self.fitres['etafit'] = fitres['etafit']
            self.fitres['aic'] = fitdiag['aic']
            if self.delta['longitude'] < 0.:  # reverse axis
                self.fitres['vlin'] = self.fitres['vlin'][::-1, :, :]
                self.fitres['hfit'] = self.fitres['hfit'][::-1, :, :]
                self.fitres['vfit'] = self.fitres['vfit'][::-1, :, :]
                self.fitres['svfit'] = self.fitres['svfit'][::-1, :, :]
                self.fitres['etafit'] = self.fitres['etafit'][::-1, :, :]
                self.fitres['aic'] = self.fitres['aic'][::-1, :]

    def pix2coord(self, pixel, name):

        # transform pixel value into coordinate value for a given coordinate
        coordinate = self.refval[name] + self.delta[name] * (
                pixel - self.refpix[name])
        return coordinate

    def coord2pix(self, coordinate, name):

        # transform coordinate value into pixel value for a given coordinate
        pixel = int(round(self.refpix[name] + (1. / self.delta[name]) * (
                coordinate - self.refval[name])))
        return pixel

    def getValue(self, ll, bb, vv):

        # get the value in the cube corresponding to the pixels ll (longitude),
        # bb (latitude), vv (velocity )
        vec = [0, 0, 0]
        vec[self.atlas['longitude'] - 1] = ll
        vec[self.atlas['latitude'] - 1] = bb
        vec[self.atlas['velocity'] - 1] = vv
        value = self.data[vec[2], vec[1], vec[0]]
        return value

    def getLineData(self, l, b, vmin, vmax):

        # extract the line data in agiven direction

        nbins = int(
            self.vscale * (vmax - vmin) / abs(self.delta['velocity'])) + 1
        vdir = int(self.delta['velocity'] / abs(self.delta['velocity']))

        vel = np.array([])
        Tb = np.array([])

        ll = self.coord2pix(l, 'longitude')
        bb = self.coord2pix(b, 'latitude')
        vvmin = self.coord2pix(self.vscale * vmin, 'velocity')
        for s in range(nbins):
            vv = int(vvmin + vdir * s)
            vel = np.append(vel, self.pix2coord(vv, 'velocity'))
            val = self.getValue(ll, bb, vv)
            Tb = np.append(Tb, val)

        vel /= self.vscale

        return vel, Tb

    def find_anomalies(self,T_thresh, outfilename):

        idx = np.where(self.data < T_thresh)

        f = open(outfilename,'w')

        N = np.shape(idx)[1]
        for s in range(N):
            lon = self.pix2coord(idx[3- self.atlas['longitude']][s],'longitude')
            lat = self.pix2coord(idx[3 - self.atlas['latitude']][s], 'latitude')
            vel = self.pix2coord(idx[3 - self.atlas['velocity']][s], 'velocity')/self.vscale
            f.write('{},{},{}\n'.format(lon,lat,vel))

        f.close()

    def getFitResults(self, l, b, vmin, vmax):

        nbins = int(
            self.vscale * (vmax - vmin) / abs(self.delta['velocity'])) + 1
        vdir = int(self.delta['velocity'] / abs(self.delta['velocity']))
        il = self.coord2pix(l, 'longitude')
        ib = self.coord2pix(b, 'latitude')
        vvmin = self.coord2pix(self.vscale * vmin, 'velocity')

        vel = np.array([])
        for s in range(nbins):
            vv = int(vvmin + vdir * s)
            vel = np.append(vel, self.pix2coord(vv, 'velocity'))
        vel /= self.vscale

        Tfit = np.zeros(nbins)
        nlin = np.sum(self.fitres['hfit'][il, ib, :] != 0.)

        PV = np.zeros((nbins, nlin)).astype('float32')
        if nlin != 0:
            for klin in range(nlin):
                PV[:, klin] = pseudoVoigt(self.fitres['hfit'][il, ib, klin],
                                          self.fitres['vfit'][il, ib, klin],
                                          self.fitres['svfit'][il, ib, klin],
                                          self.fitres['etafit'][il, ib, klin], vel)

            Tfit = np.sum(PV, axis=1).astype('float32')

        aic = self.fitres['aic'][il, ib]

        return vel, self.fitres['vfit'][il, ib, :], PV, Tfit, aic


    def line(self, l, b, vmin, vmax, vcuts=False, dcuts=False, cutfile = False,
             plotFit=False, lineDtc=False, lng=2, lis=1, sig=2.5, thresh=3., fitLine=False):

        vel, Tb = self.getLineData(l, b, vmin, vmax)

        self.ax = plt.subplot(111)
        self.ax.plot(vel, Tb, linewidth=0, color='k', marker='o', markersize=3)
        self.ax.set_xlabel('$V_\mathrm{LSR}$ (km s$^{-1}$)')
        self.ax.set_ylabel('$T_\mathrm{B}$ (K)')

        if vcuts:
            for s, vrange in enumerate(vcuts):
                lon = l
                lat = b
                vmin = eval(vrange[0])
                vmax = eval(vrange[1])
                plt.axvline(vmin, color='k')
                plt.axvline(vmax, color='k')
        elif dcuts:
            for bound in dcuts:
                lon = l
                lat = b
                vlsr = lbd2vlsr(lon, lat, bound,R0,V0,rotcurve)
                plt.axvline(vlsr, color='k')
        elif cutfile:
            cuts = np.load(cutfile)
            for cut in cuts:
                # make sure x-values are sorted
                idx = np.argsort(cut[0])
                cut[1] = cut[1][idx]
                cut[0] = cut[0][idx]
                # plot at line position
                vlsr = np.interp(l, cut[0], cut[1],R0,V0,rotcurve)
                plt.axvline(vlsr, color='k')

        if plotFit:
            if self.fitres['available']:
                vel, vfit, PV, Tfit, aic = self.getFitResults(l, b, vmin, vmax)
                for klin in range(np.shape(PV)[1]):
                    self.ax.plot(vel, PV[:, klin], color='g', linestyle='--')
                self.ax.plot(vel, Tfit, color='r')
                dev = np.sum(np.abs(Tb - Tfit)) / np.sum(Tb)
                print('AIC', aic)
                print('integrated fractional model deviation', dev)
            else:
                print("Fit results not available")

        if lineDtc:
            ilin, eps = epsDetect(Tb, lis=lis, lng=lng, sig=sig)
            ilin = np.array(ilin)
            eps = np.array(eps)
            ilin = ilin[eps > thresh]
            eps = eps[eps > thresh]
            for ii in range(len(ilin)):
                self.ax.plot(vel[ilin[ii]], eps[ii], marker='o', color='b', linewidth=0)

        if fitLine:
            fitres, model, ind_lines, vlin = self.mPSV_profile_fit(vel, Tb, lis=lis, lng=lng,
                                                                   thresh=thresh, sig=sig)
            self.ax.plot(vel, model, color='r', )
            for n in range(len(ind_lines)):
                self.ax.plot(vel, ind_lines[n], color='g', linestyle='--')
                dev = np.sum(np.abs(Tb - model)) / np.sum(Tb)
            if (fitres['is_valid'] == True or \
                (fitres['has_covariance'] == True and fitres[
                    'has_valid_parameters'] == True and \
                 (fitres['has_reached_call_limit'] == False or fitres[
                     'is_above_max_edm'] == False)) \
                    ) \
                    and dev < 1.:
                print('fit succeeded')
            else:
                print('fit failed')
                print(fitres)
            print('integrated fractional model deviation', dev)

        plt.show()

    def column(self, vel, Tb, Tbkg=2.66):

        # default Tbkg 2.66 K, CMB brightness temperature at 1.4GHz

        if self.Ts == -10.:
            intensity = self.int2col * np.sum(Tb) * np.abs(
                self.delta['velocity'])
        else:
            try:
                Tb[Tb > self.Ts - 5.] = self.Ts - 5.
                intensity = -self.int2col * np.abs(
                    self.delta['velocity']) * self.Ts * np.sum(
                    np.log(1 - Tb / (self.Ts - Tbkg)))
            except:
                intensity = -5000

        intensity /= self.vscale

        return intensity

    def mapheader(self, hdu, lmax, bmin, bunit):

        # add the map keywords to the header
        hdu.header['CRPIX1'] = 1.0
        hdu.header['CRVAL1'] = lmax
        hdu.header['CDELT1'] = -abs(self.delta['longitude'])
        hdu.header['CTYPE1'] = 'GLON-CAR'
        crpix_2 = self.coord2pix(0., 'latitude') - self.coord2pix(bmin, 'latitude') + 1
        hdu.header['CRPIX2'] = crpix_2
        hdu.header['CRVAL2'] = 0.
        hdu.header['CDELT2'] = abs(self.delta['latitude'])
        hdu.header['CTYPE2'] = 'GLAT-CAR'
        hdu.header['BUNIT'] = (bunit['unit'], bunit['quantity'])

    def commheader(self, hdu, comment):

        # add useful comments to the header
        hdu.header.add_comment(comment)

    def history(self, hdu, name, email):

        # add history cards
        hdu.header.add_history('map generated by {}, {}'.format(name, email))
        hdu.header.add_history('on ' + time.ctime() + ' ' + time.tzname[1])

    def lbmaps(self, lmin, lmax, bmin, bmax, vmin, vmax, names, vcuts=False, dcuts=False, cutfile = False,
               outdir='./', name_tag='', saveMaps=False, display=True, authname='L. Tibaldo',
               authemail='luigi.tibaldo@irap.omp.eu', useFit=False, dev_thresh=0.3):

        # check if required region is covered by input file, otherwise modify boundaries
        l1 = self.pix2coord(0, 'longitude')
        l2 = self.pix2coord(self.naxis['longitude'] - 1, 'longitude')
        ll = np.minimum(l1, l2)
        lu = np.maximum(l1, l2)
        lmin = np.maximum(lmin, ll)
        lmax = np.minimum(lmax, lu)
        b1 = self.pix2coord(0, 'latitude')
        b2 = self.pix2coord(self.naxis['latitude'] - 1, 'latitude')
        bl = np.minimum(b1, b2)
        bu = np.maximum(b1, b2)
        bmin = np.maximum(bmin, bl)
        bmax = np.minimum(bmax, bu)

        if vcuts == False and dcuts == False and cutfile == False:
            raise ValueError("Bounds for map generation not specified")
        else:
            lbins = int((lmax - lmin) / abs(self.delta['longitude'])) + 1
            bbins = int((bmax - bmin) / abs(self.delta['latitude'])) + 1
            ldir = self.delta['longitude'] / abs(self.delta['longitude'])
            bdir = self.delta['latitude'] / abs(self.delta['latitude'])

            F = plt.figure(1, (9, 8))
            F.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)

            if vcuts:
                nn = len(vcuts)
            elif dcuts:
                nn = len(dcuts) + 1
            elif cutfile:
                cuts = np.load(cutfile)
                nn = len(cuts) + 1
                # make sure x-values are sorted
                for cut in cuts:
                    idx = np.argsort(cut[0])
                    cut[1] = cut[1][idx]
                    cut[0] = cut[0][idx]

            ngrid = int(np.ceil(np.sqrt(nn)))

            grid = AxesGrid(F, 111,
                            nrows_ncols=(ngrid, ngrid),
                            axes_pad=0.2,
                            label_mode="L",
                            share_all=True,
                            cbar_location="top",
                            cbar_mode="each",
                            cbar_size="7%",
                            cbar_pad="2%",
                            )

            extent = (lmax, lmin, bmin, bmax)

            vmaps = np.zeros([nn, bbins, lbins])
            history = []

            for ll in range(lbins):
                for bb in range(bbins):
                    lpix = self.coord2pix(lmax, 'longitude') - ll * ldir
                    bpix = self.coord2pix(bmin, 'latitude') + bb * bdir
                    lon = self.pix2coord(lpix, 'longitude')
                    lat = self.pix2coord(bpix, 'latitude')
                    # if using distance cuts turn them in velocity
                    if dcuts:
                        vlsr = lbd2vlsr(lon, lat, np.array(dcuts),R0,V0,rotcurve)
                        vlsr = np.append(vmin, vlsr)
                        vlsr = np.append(vlsr, vmax)
                    # if using a file build the velocity array for the l,b bin
                    if cutfile:
                        vlsr = np.array([])
                        vlsr = np.append(vlsr,vmin)
                        for cut in cuts:
                            vval = np.interp(lon,cut[0],cut[1])
                            vlsr = np.append(vlsr,vval)
                        vlsr = np.append(vlsr, vmax)
                    # retrieve data, and, in case fit
                    vel, Tb = self.getLineData(lon, lat, vmin, vmax)
                    if useFit:
                        good_fit = False
                        velf, vfit, PV, Tfit, aic = self.getFitResults(lon, lat, vmin, vmax)
                        dev = np.sum(np.abs(Tb - Tfit)) / np.sum(Tb)
                        if np.sum(np.abs(Tb)) == 0.:
                            msg = 'lon {} lat {} NODATA'.format(lon, lat)
                            history.append(msg)
                        elif len(vfit) == 0:
                            msg = 'lon {} lat {} fit FAILED'.format(lon, lat)
                            history.append(msg)
                        elif np.abs(dev) > dev_thresh:
                            msg = 'lon {} lat {} fit BAD, integrated fractional model deviation {}'.format(
                                lon, lat, dev)
                            history.append(msg)
                        else:
                            good_fit = True
                            msg = 'lon {} lat {} integrated fractional model deviation {}'.format(
                                lon, lat, dev)
                            history.append(msg)
                    for s in range(nn):
                        if vcuts:
                            vrange = vcuts[s]
                            vlow = eval(vrange[0])
                            vup = eval(vrange[1])
                        elif dcuts or cutfile:
                            vlow = vlsr[s]
                            vup = vlsr[s + 1]
                        if useFit and good_fit:
                            # add integral of all lines that have a peak in the velo range
                            for klin, vlin in enumerate(vfit):
                                if vlin >= vlow and vlin < vup:
                                    vmaps[s, bb, ll] += self.column(velf, PV[:, klin])
                                else:
                                    pass
                            # correct for the residual colmn density
                            correction = self.column(vel[(vel >= vlow) & (vel < vup)],
                                                     Tb[(vel >= vlow) & (vel < vup)])
                            correction -= self.column(velf[(velf >= vlow) & (velf < vup)],
                                                      Tfit[(velf >= vlow) & (velf < vup)])
                            vmaps[s, bb, ll] += correction
                        else:
                            vmaps[s, bb, ll] = self.column(vel[(vel >= vlow) & (vel < vup)],
                                                           Tb[(vel >= vlow) & (vel < vup)])

            # display and in case save maps
            if saveMaps:
                # history
                # takes a while to write in fits, dump separately
                histfile = gzip.open(outdir + 'lbmap_' + name_tag + 'history.txt.gz', 'wb')
                for k, line in enumerate(history):
                    if not k == (len(history) - 1):
                        histfile.write((line + '\n').encode())
                histfile.close()

            for s in range(nn):
                im = grid[s].imshow(vmaps[s], extent=extent, interpolation='none',
                                    origin='lower', vmin=-5., cmap='Spectral_r')
                grid.cbar_axes[s].colorbar(im)
                t = add_inner_title(grid[s], names[s], loc=2)
                t.patch.set_ec("none")
                t.patch.set_alpha(0.5)
                if saveMaps:
                    try:
                        maphdu = fits.PrimaryHDU(vmaps[s])
                        lmax_out = self.pix2coord(self.coord2pix(lmax, 'longitude'), 'longitude')
                        bmin_out = self.pix2coord(self.coord2pix(bmin, 'latitude'), 'latitude')
                        bunit = {}
                        if self.int2col == 1:
                            bunit['unit'] = 'K km s-1'
                            bunit['quantity'] = 'v-integrated Tb'
                        else:
                            bunit['unit'] = 'cm-2'
                            bunit['quantity'] = 'N(H)'
                        self.mapheader(maphdu, lmax_out, bmin_out, bunit)
                        # comments
                        if self.int2col != 1.:
                            msg = 'Integral to column: {} cm-2 (K km s-1)-1'.format(self.int2col)
                            self.commheader(maphdu, msg)
                        if self.Ts != -10:
                            self.commheader(maphdu, 'Spin temperature: {} K'.format(self.Ts))
                        if vcuts:
                            self.commheader(maphdu, 'velocity cuts: ' + str(dcuts))
                        elif dcuts:
                            self.commheader(maphdu, 'heliocentric distance cuts: ' + str(dcuts))
                        elif cutfile:
                            self.commheader(maphdu, 'velocity cuts from file: ' + str(cutfile))
                        if useFit:
                            self.commheader(maphdu, 'correction based on line profile fitting')
                        self.commheader(maphdu, 'Map: n. {}, {}'.format(s, names[s]))
                        self.history(maphdu, authname, authemail)
                        maphdu.writeto(outdir + 'lbmap_' + name_tag + names[s] + '.fits')
                    except:
                        print("Saving map {} failed".format(s))

            grid.axes_llc.set_xlabel('$l$ (deg)')
            grid.axes_llc.set_ylabel('$b$ (deg)')
            grid.axes_llc.set_xlim(lmax, lmin)
            grid.axes_llc.set_ylim(bmin, bmax)

            if display:
                plt.show()
            else:
                pass


    def vdiagram(self, lmin, lmax, bmin, bmax, vmin, vmax, integrate='latitude'):

        # convert boundaries into pixels
        imin = self.coord2pix(lmin, 'longitude')
        imax = self.coord2pix(lmax, 'longitude')
        jmin = self.coord2pix(bmin, 'latitude')
        jmax = self.coord2pix(bmax, 'latitude')
        kmin = self.coord2pix(vmin * self.vscale, 'velocity')
        kmax = self.coord2pix(vmax * self.vscale, 'velocity')
        # establish sense of increasing longitude and velocity
        ldir = self.delta['longitude'] / abs(self.delta['longitude'])
        vdir = self.delta['velocity'] / abs(self.delta['velocity'])

        # set boundaries according to axes order and orientation
        pixmin = [0, 0, 0]
        pixmax = [0, 0, 0]
        if ldir > 0:
            pixmin[self.atlas['longitude'] - 1] = imin
            pixmax[self.atlas['longitude'] - 1] = imax
        else:
            pixmin[self.atlas['longitude'] - 1] = imax
            pixmax[self.atlas['longitude'] - 1] = imin
        pixmin[self.atlas['latitude'] - 1] = jmin
        pixmax[self.atlas['latitude'] - 1] = jmax
        if vdir > 0:
            pixmin[self.atlas['velocity'] - 1] = kmin
            pixmax[self.atlas['velocity'] - 1] = kmax
        else:
            pixmin[self.atlas['velocity'] - 1] = kmax
            pixmax[self.atlas['velocity'] - 1] = kmin
        im = self.data[pixmin[2]:pixmax[2], pixmin[1]:pixmax[1], pixmin[0]:pixmax[0]]

        # if we integrate over latitude make sure longitude increases right to left
        if integrate == 'latitude' and ldir > 0:
            im = np.flip(im, axis=(3 - self.atlas['longitude']))
        if vdir < 0:
            im = np.flip(im, axis=(3 - self.atlas['velocity']))
        # always make sure velocity increases bottom to top
        # integrate over appropriate axis
        im = np.sum(im, axis=(3 - self.atlas[integrate]))

        # multiply by Delta to obtain K deg
        if integrate == 'latitude':
            im *= np.abs(self.delta['latitude'])
        elif integrate == 'longitude':
            im *= np.abs(self.delta['longitude'])

        # create the figure
        ax = plt.subplot(111)

        # reorder so that axes appear in the "right" place
        # and set figure extent and axis labels
        if integrate == 'latitude':
            if self.atlas['velocity'] < self.atlas['longitude']:
                im = im.transpose()
            extent = (lmax, lmin, vmin, vmax)
            ax.set_xlabel('$l$ (deg)')
            ax.set_ylabel('V (km s$^{-1}$)')
        elif integrate == 'longitude':
            if self.atlas['velocity'] > self.atlas['latitude']:
                im = im.transpose()
            extent = (vmin, vmax, bmin, bmax)
            ax.set_xlabel('V (km s$^{-1}$)')
            ax.set_ylabel('$b$ (deg)')

        # display the map
        plt.imshow(im, interpolation='none', origin='lower', extent=extent, aspect='auto',
                   norm=LogNorm(vmin=2*np.abs(self.delta['latitude'])), cmap='jet')
        cbar = plt.colorbar(label="K deg")

        plt.show()


    def vdiagram_fit(self, lmin, lmax, bmin, bmax, vmin, vmax, integrate='latitude'):

        # check if required region is covered by input file, otherwise modify boundaries
        l1 = self.pix2coord(0, 'longitude')
        l2 = self.pix2coord(self.naxis['longitude'] - 1, 'longitude')
        ll = np.minimum(l1, l2)
        lu = np.maximum(l1, l2)
        lmin = np.maximum(lmin, ll)
        lmax = np.minimum(lmax, lu)
        b1 = self.pix2coord(0, 'latitude')
        b2 = self.pix2coord(self.naxis['latitude'] - 1, 'latitude')
        bl = np.minimum(b1, b2)
        bu = np.maximum(b1, b2)
        bmin = np.maximum(bmin, bl)
        bmax = np.minimum(bmax, bu)
        v1 = self.pix2coord(0, 'velocity')/self.vscale
        v2 = self.pix2coord(self.naxis['velocity'] - 1, 'velocity')/self.vscale
        vl = np.minimum(v1, v2)
        vu = np.maximum(v1, v2)
        vmin = np.maximum(vmin, vl)
        vmax = np.minimum(vmax, vu)

        # binning parameters
        lbins = int((lmax - lmin) / abs(self.delta['longitude'])) + 1
        bbins = int((bmax - bmin) / abs(self.delta['latitude'])) + 1
        vbins = int((vmax - vmin) / abs(self.delta['velocity']/self.vscale)) + 1
        ldir = self.delta['longitude'] / abs(self.delta['longitude'])
        bdir = self.delta['latitude'] / abs(self.delta['latitude'])
        vdir = self.delta['velocity'] / abs(self.delta['velocity'])

        # create output array
        if integrate == 'latitude':
            im = np.zeros([vbins,lbins])
        elif integrate == 'longitude':
            im = np.zeros([bbins, vbins])

        for ll in range(lbins):
            print(ll,'of',lbins)
            for bb in range(bbins):
                lpix = self.coord2pix(lmax, 'longitude') - ll * ldir
                bpix = self.coord2pix(bmin, 'latitude') + bb * bdir
                lon = self.pix2coord(lpix, 'longitude')
                lat = self.pix2coord(bpix, 'latitude')
                velf, vfit, PV, Tfit, aic = self.getFitResults(lon, lat, vmin, vmax)
                for vv in range(vbins):
                    vpix = self.coord2pix(self.vscale * vmin, 'velocity') + vv * vdir
                    vel = self.pix2coord(vpix, 'velocity')/self.vscale
                    for klin, vlin in enumerate(vfit):
                        # check if line velocity falls within bin and that there is valid a fitted profile
                        if np.abs(vlin - vel) <= np.abs(self.delta['velocity']/self.vscale) / 2 and \
                                klin < len(PV[0,:]):
                            if integrate == 'latitude':
                                im[vv,ll] += self.column(velf, PV[:, klin])
                            elif integrate == 'longitude':
                                im[bb,vv] += self.column(velf, PV[:, klin])
                        else:
                            pass

        print('done')

        # create the figure
        ax = plt.subplot(111)

        # and set figure extent and axis labels
        if integrate == 'latitude':
            extent = (lmax, lmin, vmin, vmax)
            ax.set_xlabel('$l$ (deg)')
            ax.set_ylabel('V (km s$^{-1}$)')
        if integrate == 'longitude':
            extent = (vmin, vmax, bmin, bmax)
            ax.set_xlabel('V (km s$^{-1}$)')
            ax.set_ylabel('$b$ (deg)')

        # display the map
        plt.imshow(im,
                   interpolation='none',
                   origin='lower', extent=extent, aspect='auto',
                   norm=LogNorm(vmin=1),
                   cmap='jet')
        cbar = plt.colorbar(label="N(H) 10^20 cm^-2")

        plt.show()

    def weightedv_maps(self, lmin, lmax, bmin, bmax, vmin, vmax, threshold=3.,
                       cmap = 'Spectral_r',display=True):

        # check if required region is covered by input file, otherwise modify boundaries
        l1 = self.pix2coord(0, 'longitude')
        l2 = self.pix2coord(self.naxis['longitude'] - 1, 'longitude')
        ll = np.minimum(l1, l2)
        lu = np.maximum(l1, l2)
        lmin = np.maximum(lmin, ll)
        lmax = np.minimum(lmax, lu)
        b1 = self.pix2coord(0, 'latitude')
        b2 = self.pix2coord(self.naxis['latitude'] - 1, 'latitude')
        bl = np.minimum(b1, b2)
        bu = np.maximum(b1, b2)
        bmin = np.maximum(bmin, bl)
        bmax = np.minimum(bmax, bu)

        lbins = int((lmax - lmin) / abs(self.delta['longitude'])) + 1
        bbins = int((bmax - bmin) / abs(self.delta['latitude'])) + 1
        ldir = self.delta['longitude'] / abs(self.delta['longitude'])
        bdir = self.delta['latitude'] / abs(self.delta['latitude'])

        maps = np.zeros([2,bbins, lbins])

        for ll in range(lbins):
            for bb in range(bbins):
                lpix = self.coord2pix(lmax, 'longitude') - ll * ldir
                bpix = self.coord2pix(bmin, 'latitude') + bb * bdir
                lon = self.pix2coord(lpix, 'longitude')
                lat = self.pix2coord(bpix, 'latitude')
                # retrieve data
                vel, Tb = self.getLineData(lon, lat, vmin, vmax)
                if np.max(Tb) > threshold:
                    maps[0, bb, ll] = np.sum(vel*Tb)/np.sum(Tb)
                else:
                    maps[0 ,bb, ll] = np.nan
                maps[1, bb, ll] = np.sum(Tb)

        maps[1] *= np.abs(self.delta['velocity'])/self.vscale

        # create the figure
        F = plt.figure(1, (9, 7))
        F.subplots_adjust(left=0.12, right=0.9, top=0.95, bottom=0.08)
        grid = AxesGrid(F, 111,
                        nrows_ncols=(2, 1),
                        axes_pad=0.2,
                        label_mode="L",
                        share_all=True,
                        cbar_location="right",
                        cbar_mode="each",
                        cbar_size="7%",
                        cbar_pad="2%",
                        )

        extent = (lmax, lmin, bmin, bmax)

        # display the map
        for s in range(2):
            im = grid[s].imshow(maps[s],
                                interpolation='none',
                                origin='lower', cmap=cmap, extent=extent)
            if s==0:
                grid.cbar_axes[s].colorbar(im,label="V (km s$^{-1}$)")
            elif s==1:
                grid.cbar_axes[s].colorbar(im,label="W$_\mathrm{CO}$ (K km s$^{-1}$)")

        # and set figure extent and axis labels
        grid.axes_llc.set_xlabel('Galactic longitude (deg)')
        grid.axes_llc.set_ylabel('Galactic latitude (deg)')
        grid.axes_llc.set_xlim(lmax, lmin)
        grid.axes_llc.set_ylim(bmin, bmax)

        if display:
            plt.show()

        return F, grid