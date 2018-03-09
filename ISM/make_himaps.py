import sys
import yaml
from gascube import gascube

configfile = sys.argv[1]

with open(configfile, "r") as f:
    config = yaml.load(f)

try:
    Ts = config['Ts']
except:
    Ts = -10

for s in range(len(config['infile'])):
    cube = gascube(config['infile'][s], int2col=1.823e-2, Ts=Ts,
                   fitres_files=[config['fitres'][s],
                                 config['fitdiag'][s]])
    cube.lbmaps(config['lmin'], config['lmax'],
                config['bmin'], config['bmax'],
                config['vmin'], config['vmax'],
                list(config['names'].values()),
                dcuts=list(config['dcuts'].values()),
                saveMaps=True, useFit=config['useFit'],
                outdir=config['outdir'])
