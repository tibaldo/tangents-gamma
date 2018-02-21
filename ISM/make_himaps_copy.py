import sys
import yaml
from gascube import gascube

configfile = sys.argv[1]

with open(configfile, "r") as f:
    config = yaml.load(f)

for s in range(len(config['infile'])):
    cube = gascube(config['infile'][s], int2col=1.823e-2) # Constant to increase the visibility (contrast)
    cube.lbmaps(config['lmin'], config['lmax'], #dont use _fit for fast computation
                config['bmin'], config['bmax'],
                config['vmin'], config['vmax'],
                list(config['names'].values()),
                dcuts = list(config['dcuts'].values()),
                saveMaps=True,
                outdir=config['outdir'])
