import sys
sys.path.append('../ISM/')
from base_atlas import make_atlas

# parameters
border = 3.
lmin = 46.
lmax = 53.5
nbounds = 3
bfilename = 'Sgr_bound.npy'

make_atlas(lmin,lmax,border,nbounds,bfilename)