import sys
import yaml
from gascube import gascube

configfile = '../Sgr/E02_himaps.yaml'

with open(configfile, "r") as f:
    config = yaml.load(f)

for s in range(len(config['infile'])):
    cube = gascube(config['infile'][s])
    cube.lbmaps(config['lmin'], config['lmax'], #dont use _fit for fast computation
                config['bmin'], config['bmax'],
                config['vmin'], config['vmax'],
                list(config['names'].values()),
                dcuts = list(config['dcuts'].values()),
                saveMaps=True,
                outdir=config['outdir'])