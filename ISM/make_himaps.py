import sys
import yaml
from gascube import gascube
from map_utils import *
from multiprocessing import Pool

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
    name_tag = [''] * len(config['infile'])

try:
    nthread = config['nthread']
except:
    nthread = 1


# parallelize with multiprocessing

def create_maps(infile, Ts, fitres_files,
                lmin, lmax, bmin, bmax, vmin, vmax,
                names, cutfile, name_tag,
                useFit, outdir,
                T_anomaly, anomaly_file):
    cube = gascube(infile, int2col=1.823e-2, Ts=Ts,
                   fitres_files=fitres_files)
    cube.lbmaps(lmin, lmax, bmin, bmax, vmin, vmax,
                names, cutfile=cutfile, name_tag=name_tag,
                useFit=useFit,
                outdir=outdir,
                saveMaps=True, display=False)
    # cube.find_anomalies(T_anomaly, anomaly_file)


lmin = config['lmin']
lmax = config['lmax']
bmin = config['bmin']
bmax = config['bmax']
vmin = config['vmin']
vmax = config['vmax']
names = list(config['names'].values())
cutfile = config['cutfile']
useFit = config['useFit']
outdir = config['outdir']
T_anomaly = config['T_anomaly']

if nthread > 1:
    pool = Pool(processes=min(len(config['infile']),nthread))

for s in range(len(config['infile'])):
    try:
        fitres_files = [config['fitres'][s], config['fitdiag'][s]]
    except:
        fitres_files = [None, None]
    infile = config['infile'][s]
    anomaly_file = outdir + '/{}anomalies.dat'.format(name_tag[s])
    args = (infile, Ts, fitres_files,
            lmin, lmax, bmin, bmax, vmin, vmax,
            names, cutfile, name_tag[s],
            useFit,
            outdir,
            T_anomaly, anomaly_file)
    if nthread > 1:
        pool.apply_async(create_maps, args)
    else:
        create_maps(*args)

if nthread > 1:
    pool.close()
    pool.join()

# target_res = config['target_res']
#
# if nthread > 1:
#     pool = Pool(processes=min(len(dcuts)+1,nthread))
#
# for ireg in range(len(dcuts)+1):
#     # merge maps
#     filenames = [outdir + 'lbmap_' + name_tag[s] + names[ireg] + '.fits'
#                  for s in range(len(config['infile']))]
#     mergedfile = outdir + 'lbmap_merged_' + names[ireg] + '.fits'
#     args = (filenames,mergedfile,
#             target_res,
#             lmin,lmax,bmin,bmax,vmin,vmax,
#             dcuts,ireg)
#     if nthread > 1:
#         pool.apply_async(merge_maps, args)
#     else:
#         merge_maps(*args)
#
# if nthread > 1:
#     pool.close()
#     pool.join()

