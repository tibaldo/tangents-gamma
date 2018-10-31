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

try:
    name_tag = config['name_tag']
except:
    name_tag = ''

for s in range(len(config['infile'])):
    try:
        fitres_files=[config['fitres'][s],config['fitdiag'][s]]
    except:
        fitres_files=[None,None]
    cube = gascube(config['infile'][s], int2col=1.823e-2, Ts=Ts,
                   fitres_files=fitres_files)
    cube.lbmaps(config['lmin'], config['lmax'],
                config['bmin'], config['bmax'],
                config['vmin'], config['vmax'],
                list(config['names'].values()),
                dcuts=list(config['dcuts'].values()),
                saveMaps=True, useFit=config['useFit'],
                outdir=config['outdir'], name_tag = name_tag)
