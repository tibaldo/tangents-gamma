from astropy.io import fits
import numpy as np


class gascube:
    def __init__(self, filename):

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
                    'CTYPE' + str(i + 1)) == 'VEL'):
                self.atlas['velocity'] = i + 1
                self.refpix['velocity'] = self.header.get(
                    'CRPIX' + str(i + 1)) - 1
                self.refval['velocity'] = self.header.get('CRVAL' + str(i + 1))
                self.delta['velocity'] = self.header.get('CDELT' + str(i + 1))
                # store velocity unit
                self.is_km = True
                try:
                    u = self.header.get('CUNIT' + str(i + 1))
                    if u == 'M/S':
                        self.is_km = False
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
            print("ERROR, anomalous number of axes in fits file", filename)
        self.data[self.data <= (blankvalue + 0.1)] = 0.
        self.data = np.nan_to_num(self.data)
