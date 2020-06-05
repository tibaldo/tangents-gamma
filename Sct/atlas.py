import sys
sys.path.append('../ISM/')
from base_atlas import make_atlas

# parameters
border = 3.
lmin = 26.
lmax = 35
nbounds = 3
bfilename = 'Sct_bound.npy'
hifilename = "/Users/ltibaldo/Fermi/ISM/HI/HI4PI/CAR_E02.fits"

make_atlas(lmin,lmax,border,nbounds,bfilename,hifilename,nin=3)