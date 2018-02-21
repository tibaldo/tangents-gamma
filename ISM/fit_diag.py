from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]

header=fits.getheader(filename)

record=header["RECORD"]
lines=record.split('/')

dev=np.array([])
nodata=0
failed=0
badfit=0


for line in lines:
    if 'NODATA' in line:
        nodata+=1
    elif 'FAILED' in line:
        failed+=1
    elif 'BAD' in line:
        badfit+=1
    else:
        if float(line.split('deviation')[1])>1:
            print(line)
        dev=np.append(dev,float(line.split('deviation')[1]))

ax=plt.subplot(111)

n, bins, patches = plt.hist(dev, 50, normed=1, histtype='step')

ax.set_xlabel("integrated absolute deviation (fraction)")

plt.text(0.6, 0.7,
         "nodata: {}\nfailed: {}\nbad fit: {}\nsucceeded: {}\n    average = {:.3f}\n    median = {:.3f}\n    RMS = {:.3f}".format(nodata,failed,badfit,len(dev),np.average(dev),np.median(dev),np.std(dev)),
         transform = ax.transAxes, bbox={'boxstyle':'Round','facecolor':'white', 'alpha':0.2})

plt.show()
    
