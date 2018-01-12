from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from iminuit import Minuit
import time

from reid14_cordes02 import lbd2vlsr
from epsDetectS import epsDetect
from mPSV import multiPSV_chi2


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
    def __init__(self, filename, int2col=1., Ts=-10):

        hdus = fits.open(filename)

        # store header
        self.header = hdus[0].header

        # read the axis type and mapping values
        naxis = self.header.get('NAXIS')
        self.atlas = {}
        self.refpix = {}
        self.refval = {}
        self.delta = {}
        for i in range(naxis):
            if (self.header.get('CTYPE' + str(i + 1)) == 'GLON-CAR'):
                self.atlas['longitude'] = i + 1
                self.refpix['longitude'] = self.header.get(
                    'CRPIX' + str(i + 1)) - 1
                self.refval['longitude'] = self.header.get('CRVAL' + str(i + 1))
                self.delta['longitude'] = self.header.get('CDELT' + str(i + 1))
            if (self.header.get('CTYPE' + str(i + 1)) == 'GLAT-CAR'):
                self.atlas['latitude'] = i + 1
                self.refpix['latitude'] = self.header.get(
                    'CRPIX' + str(i + 1)) - 1
                self.refval['latitude'] = self.header.get('CRVAL' + str(i + 1))
                self.delta['latitude'] = self.header.get('CDELT' + str(i + 1))
            if (self.header.get(
                        'CTYPE' + str(i + 1)) == 'VELO-LSR' or self.header.get(
                    'CTYPE' + str(i + 1)) == 'VELO-LSRK' or self.header.get(
                    'CTYPE' + str(i + 1)) == 'VEL' or self.header.get(
                    'CTYPE' + str(i + 1)) == 'VRAD'):
                self.atlas['velocity'] = i + 1
                self.refpix['velocity'] = self.header.get(
                    'CRPIX' + str(i + 1)) - 1
                self.refval['velocity'] = self.header.get('CRVAL' + str(i + 1))
                self.delta['velocity'] = self.header.get('CDELT' + str(i + 1))
                # store velocity unit
                self.vscale = 1.
                try:
                    u = self.header.get('CUNIT' + str(i + 1))
                    if u == 'M/S' or u == 'm/s':
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
        self.data[self.data <= (blankvalue + 0.1)] = 0.
        self.data = np.nan_to_num(self.data)

        self.int2col = int2col
        self.Ts = Ts  # default=-10 is optically thin approx

    def pix2coord(self, pixel, name):

        # transform pixel value into coordinate value for a given coordinate
        coordinate = self.refval[name] + self.delta[name] * (
            pixel - self.refpix[name])
        return coordinate

    def coord2pix(self, coordinate, name):

        # transform coordinate value into pixel value for a given coordinate
        pixel = round(self.refpix[name] + (1. / self.delta[name]) * (
            coordinate - self.refval[name]))
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

    def mPSV_profile_fit(self, vv, tb, lis=1, lng=2, thresh=3., sig=2.5, print_level=1):

        # line detection
        ilin, eps = epsDetect(tb, lis=lis, lng=lng, sig=sig)
        ilin = np.array(ilin).astype('int')
        eps = np.array(eps)
        ilin = ilin[eps > thresh]
        eps = eps[eps > thresh]
        vlin = vv[ilin]

        # fit, define chi square
        chi2 = multiPSV_chi2(vv, tb)
        # define params tuple, initial values, limits, etc
        ptup = ()
        kwdarg = {}
        for n in range(len(eps)):
            ptup = ptup + ('A_' + str(n),)
            kwdarg['A_' + str(n)] = eps[n]
            kwdarg['error_A_' + str(n)] = 10
            kwdarg['limit_A_' + str(n)] = (0., 1.e8)
            ptup = ptup + ('x0_' + str(n),)
            kwdarg['x0_' + str(n)] = vlin[n]
            kwdarg['error_x0_' + str(n)] = 0.5
            kwdarg['limit_x0_' + str(n)] = (vlin[n] - 5., vlin[n] + 5.)
            ptup = ptup + ('gammaG_' + str(n),)
            kwdarg['gammaG_' + str(n)] = 5.
            kwdarg['error_gammaG_' + str(n)] = 2.
            kwdarg['limit_gammaG_' + str(n)] = (0.01, 1.e2)
            ptup = ptup + ('gammaL_' + str(n),)
            kwdarg['gammaL_' + str(n)] = 5.
            kwdarg['error_gammaL_' + str(n)] = 2.
            kwdarg['limit_gammaL_' + str(n)] = (0.01, 1.e2)
            # create minuit object, minimize, return results
        m = Minuit(chi2, forced_parameters=ptup, errordef=1, print_level=print_level, **kwdarg)
        fitres = m.migrad()[0]
        model = chi2.multiPSV(*m.args)
        v_lines = []
        ind_lines = []
        for n in range(len(eps)):
            v_lines.append(m.args[4 * n + 1])
            ind_lines.append(chi2.PSV(*m.args[4 * n:4 * (n + 1)]))

        del m  # try to save memory

        return fitres, model, ind_lines, v_lines

    def line(self, l, b, vmin, vmax, vcuts=False, dcuts=False, lineDtc=False, lng=2, lis=1,
             sig=2.5, thresh=3., fitLine=False):

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

        if dcuts:
            for bound in dcuts:
                lon = l
                lat = b
                vlsr = lbd2vlsr(lon, lat, bound)
                plt.axvline(vlsr, color='k')

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

    def lbmaps(self, lmin, lmax, bmin, bmax, vmin, vmax, names, vcuts=False, dcuts=False,
               outdir='./', saveMaps=False, display=True, authname='L. Tibaldo',
               authemail='luigi.tibaldo@irap.omp.eu'):

        if vcuts == False and dcuts == False:
            raise ValueError("Bounds for map generation not specified")
        else:
            lbins = int((lmax - lmin) / abs(self.delta['longitude'])) + 1
            bbins = int((bmax - bmin) / abs(self.delta['latitude'])) + 1
            ldir = self.delta['longitude'] / abs(self.delta['longitude'])
            bdir = self.delta['latitude'] / abs(self.delta['latitude'])

            F = plt.figure(1, (9, 8))
            F.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)

            if vcuts:
                nn = len(vcuts) + 1
            elif dcuts:
                nn = len(dcuts) + 1

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

            for s in range(nn):
                vmap = np.zeros([bbins, lbins])
                for ll in range(lbins):
                    for bb in range(bbins):
                        lpix = self.coord2pix(lmax, 'longitude') - ll * ldir
                        bpix = self.coord2pix(bmin, 'latitude') + bb * bdir
                        lon = self.pix2coord(lpix, 'longitude')
                        lat = self.pix2coord(bpix, 'latitude')
                        if vcuts:
                            vrange = vcuts[s]
                            vlow = eval(vrange[0])
                            vup = eval(vrange[1])
                        elif dcuts:
                            vlsr = lbd2vlsr(lon, lat, np.array(dcuts))
                            vlsr = np.append(vmin, vlsr)
                            vlsr = np.append(vlsr, vmax)
                            vlow = vlsr[s]
                            vup = vlsr[s + 1]
                        vel, Tb = self.getLineData(lon, lat, vlow, vup)
                        vmap[bb, ll] = self.column(vel, Tb)
                im = grid[s].imshow(vmap, extent=extent, interpolation='none', origin='lower')
                grid.cbar_axes[s].colorbar(im)
                t = add_inner_title(grid[s], names[s], loc=2)
                t.patch.set_ec("none")
                t.patch.set_alpha(0.5)
                if saveMaps:
                    maphdu = fits.PrimaryHDU(vmap)
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
                    self.commheader(maphdu, 'Map: n. {}, {}'.format(s, names[s]))
                    # history
                    self.history(maphdu, authname, authemail)
                    maphdu.writeto(outdir + 'lbmap_' + names[s] + '.fits')

            grid.axes_llc.set_xlabel('$l$ (deg)')
            grid.axes_llc.set_ylabel('$b$ (deg)')
            grid.axes_llc.set_xlim(lmax, lmin)
            grid.axes_llc.set_ylim(bmin, bmax)

            if display:
                plt.show()
            else:
                pass

    def lbmaps_fit(self, lmin, lmax, bmin, bmax, vmin, vmax, names, vcuts=False, dcuts=False,
                   outdir='./', saveMaps=False, lng=2, lis=1, sig=2.5, thresh=3,
                   authname='L. Tibaldo',
                   authemail='luigi.tibaldo@irap.omp.eu'):

        if vcuts == False and dcuts == False:
            raise ValueError("Bounds for map generation not specified")
        else:
            lbins = int((lmax - lmin) / abs(self.delta['longitude'])) + 1
            bbins = int((bmax - bmin) / abs(self.delta['latitude'])) + 1
            ldir = self.delta['longitude'] / abs(self.delta['longitude'])
            bdir = self.delta['latitude'] / abs(self.delta['latitude'])

            if vcuts:
                nn = len(vcuts) + 1
            elif dcuts:
                nn = len(dcuts) + 1

            vmaps = np.zeros([nn, bbins, lbins])

            history = []
            for ll in range(lbins):
                print(ll, 'of', (lbins - 1))
                for bb in range(bbins):
                    ##### Basic quantities
                    lpix = self.coord2pix(lmax, 'longitude') - ll * ldir
                    bpix = self.coord2pix(bmin, 'latitude') + bb * bdir
                    lon = self.pix2coord(lpix, 'longitude')
                    lat = self.pix2coord(bpix, 'latitude')
                    vel, Tb = self.getLineData(lon, lat, vmin, vmax)
                    ##### Fitting
                    fit_success = True
                    if np.any(Tb <= -3) or len(Tb[Tb < -0.3]) > 30:
                        history.append('lon {} lat {} FAILED: invalid values'.format(lon, lat))
                        fit_success = False
                    else:
                        fitres, model, ind_lines, vlin = self.mPSV_profile_fit(vel, Tb,
                                                                               lis=lis,
                                                                               lng=lng,
                                                                               thresh=thresh,
                                                                               sig=sig,
                                                                               print_level=0)
                        dev = np.sum(np.abs(Tb - model)) / np.sum(Tb)
                        if (fitres['is_valid'] == True or \
                                    (fitres['has_covariance'] == True and fitres[
                                        'has_valid_parameters'] == True and \
                                             (fitres['has_reached_call_limit'] == False or
                                                      fitres['is_above_max_edm'] == False)) \
                            ) \
                                and dev < 1.:
                            msg = 'lon {} lat {} integrated fractional model deviation {}'.format(
                                lon, lat, dev)
                            history.append(msg)
                        else:
                            msg = 'lon {} lat {} FAILED: fit output {}, data-model deviation {}'.format(
                                lon, lat, fitres, dev)
                            history.append(msg)
                            fit_success = False
                        ##### Fitting
                        for s in range(nn):
                            # calculate v boundaries
                            if vcuts:
                                vrange = vcuts[s]
                                vlow = eval(vrange[0])
                                vup = eval(vrange[1])
                            elif dcuts:
                                vlsr = lbd2vlsr(lon, lat, np.array(dcuts))
                                vlsr = np.append(vmin, vlsr)
                                vlsr = np.append(vlsr, vmax)
                                vlow = vlsr[s]
                                vup = vlsr[s + 1]
                            # calculate column densities
                            if fit_success:
                                # add integral of lines belonging to region
                                for ii, ivlin in enumerate(vlin):
                                    if (ivlin >= vlow) and (ivlin < vup):
                                        vmaps[s, bb, ll] += self.column(vel, ind_lines[ii])
                                # add data/model difference
                                vv = vel[(vel >= vlow) & (vel < vup)]
                                tt = Tb[(vel >= vlow) & (vel < vup)]
                                mm = model[(vel >= vlow) & (vel < vup)]
                                vmaps[s, bb, ll] += self.column(vv, tt) - self.column(vv, mm)
                            else:
                                # just use integral
                                vv = vel[(vel >= vlow) & (vel < vup)]
                                tt = Tb[(vel >= vlow) & (vel < vup)]
                                vmaps[s, bb, ll] += self.column(vv, tt)


            if saveMaps:

                histxt = ''
                for s, line in enumerate(history):
                    if not s == (len(history) - 1):
                        histxt = histxt + line + '/'
                    else:
                        histxt = histxt + line

                for s in range(nn):
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
                    self.commheader(maphdu, 'Map: n. {}, {}'.format(s, names[s]))
                    maphdu.header["RECORD"] = histxt
                    # history
                    self.history(maphdu, authname, authemail)
                    maphdu.writeto(outdir + 'lbmap_fit_' + names[s] + '.fits')

            else:
                pass
