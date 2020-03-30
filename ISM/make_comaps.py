import sys
import yaml
from gascube import gascube

configfile = sys.argv[1]

with open(configfile, "r") as f:
    config = yaml.load(f)

for s in range(len(config['infile'])):
    cube = gascube(config['infile'][s],
                   fitres_files=[config['fitres'][s],config['fitdiag'][s]])
    cube.lbmaps(config['lmin'], config['lmax'],
                config['bmin'], config['bmax'],
                config['vmin'], config['vmax'],
                list(config['names'].values()),
                vcuts=list(config['vcuts'].values()),
                saveMaps=True, useFit=config['useFit'],
                outdir=config['outdir'])
