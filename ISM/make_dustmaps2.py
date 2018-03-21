import sys
import yaml

from dustmap_residuals import dustmap_residuals

configfile = sys.argv[1]

with open(configfile, "r") as f:
    config = yaml.load(f)

gasmaps = []
for s in range(len(config['names'])):
    maps = [config['gasmaps'][s][key] for key in config['gasmaps'][s].keys()]
    gasmaps.append(maps)

names = [config['names'][key] for key in config['gasmaps'].keys()]

resid = dustmap_residuals(config['infile'], config['colname'], gasmaps,
                          scale=config['scale'], errorname=config['errorname'])
resid.make(config['lmin'], config['lmax'], config['bmin'], config['bmax'], config['pixsize'],
           'dust_residuals', names, outdir=config['outdir'], mask=config['mask'])
