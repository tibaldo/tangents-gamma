import sys
import yaml

from extcube_residuals import extcube_residuals

configfile = sys.argv[1]

with open(configfile, "r") as f:
    config = yaml.load(f)

for s in range(len(config['names'])):
    gasmaps = [config['gasmaps'][s][key] for key in config['gasmaps'][s].keys()]
    dcuts = [[cut['min'], cut['max']] for cut in
             [config['dcuts'][s][key] for key in config['dcuts'][s].keys()]]
    resid = extcube_residuals(config['infile'], gasmaps, dcuts, scale=config['scale'])
    resid.make(42.,58.,-9.,9,0.1, 'dust_residuals_' + config['names'][s],
               outdir=config['outdir'], mask=config['mask'])
