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


error_mode = config['error_mode']
if error_mode == 'ERROR':
    try:
        errorname = config['errorname']
        error_frac = 1.
    except:
        print('You must provide an error map to use ERROR mode')
else:
    errorname = 'None'
    error_frac = config['error_frac']

try:
    mask = config['mask']
except:
    mask='None'

try:
    max_iter = config['max_iter']
    smooth_radius = config['smooth_radius']
    threshold = config['threshold']
except:
    max_iter = 1
    smooth_radius = 0.
    threshold = 0


resid = dustmap_residuals(config['infile'], config['mapname'], gasmaps,
                          scale=config['scale'], errorname=errorname,hpx=config['HPX'])
resid.make(config['lmin'], config['lmax'], config['bmin'], config['bmax'], config['pixsize'],
           'dnm', names,
           error_mode = error_mode, error_frac=error_frac,
           max_iter = max_iter, smooth_radius = smooth_radius, threshold = threshold,
           outdir=config['outdir'], mask=mask)
