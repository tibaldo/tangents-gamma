import sys
import os
import yaml
import numpy as np
from gascube import gascube
from map_utils import *
from multiprocessing import Pool

configfile = sys.argv[1]

# function to parallelize map creation with multiprocessing
def create_maps(infile, Ts, fitres_files,
                lmin, lmax, bmin, bmax, vmin, vmax,
                names, cut_method, cuts, name_tag,
                useFit, outdir,
                T_anomaly, border, anomaly_file, inpaint):
    cube = gascube(infile, int2col=1.823e-2, Ts=Ts,
                   fitres_files=fitres_files)
    ### map creation #######################################
    if cut_method == 'VELOCITY':
        cube.lbmaps(lmin, lmax, bmin, bmax, vmin, vmax,
                    names, vcuts=cuts, name_tag=name_tag,
                    useFit=useFit,
                    outdir=outdir,
                    saveMaps=True, display=False)
    elif cut_method == 'DISTANCE':
        cube.lbmaps(lmin, lmax, bmin, bmax, vmin, vmax,
                    names, dcuts=duts, name_tag=name_tag,
                    useFit=useFit,
                    outdir=outdir,
                    saveMaps=True, display=False)
    elif cut_method == 'FILE':
        cube.lbmaps(lmin, lmax, bmin, bmax, vmin, vmax,
                    names, cutfile=cuts, name_tag=name_tag,
                    useFit=useFit,
                    outdir=outdir,
                    saveMaps=True, display=False)
    ### mask/inpaint anomalies ################################
    if T_anomaly:
        # find anomalies
        cube.find_anomalies(T_anomaly, anomaly_file)
        # mask/inpaint
        for ireg, name in enumerate(names):
            filename = outdir + 'lbmap_' + name_tag + name + '.fits'
            maskfile = outdir + 'anomalymask_' + name_tag + name + '.fits'
            if cut_method == 'VELOCITY':
                mask = create_anomaly_mask(fits.getheader(filename),
                                           anomaly_file, vmin, vmax, ireg,
                                           vcuts=cuts, dcuts=False, cutfile = False,
                                           border=border,
                                           save=True, outfilename=maskfile)
            if cut_method == 'DISTANCE':
                mask = create_anomaly_mask(fits.getheader(filename),
                                           anomaly_file, vmin, vmax, ireg,
                                           vcuts=False, dcuts=True, cutfile = False,
                                           border=border,
                                           save=True, outfilename=maskfile)
            if cut_method == 'FILE':
                mask = create_anomaly_mask(fits.getheader(filename),
                                           anomaly_file, vmin, vmax, ireg,
                                           vcuts=False, dcuts=False, cutfile = cuts,
                                           border = border,
                                           save=True, outfilename=maskfile)
            if not mask=='NONE' and inpaint:
                inptfile = outdir + 'lbmap_inpainted_' + name_tag + name + '.fits'
                inpaint_map(filename,mask,inptfile)


# load parameters
with open(configfile, "r") as f:
    config = yaml.load(f)

# optional parameters

#Ts
try:
    Ts = config['Ts']
except:
    Ts = -10

#name tags for different datasets
try:
    name_tag = config['name_tag']
except:
    name_tag = [''] * len(config['infile'])

#number of parallel threads
try:
    nthread = config['nthread']
except:
    nthread = 1

#search for anomalies due to radio sources
try:
    T_anomaly = config['T_anomaly']
except:
    T_anomaly = False

# add border to mask?
# inpaint over anomalies?
border = 0
inpaint = False
if not T_anomaly:
    pass
else:
    try:
        inpaint = config['inpaint']
        border = config['border']
    except:
        inpaint = False

#use fit for los separation?
try:
    useFit = config['useFit']
except:
    useFit = False

###### determine cut method
cut_method = 'NONE'
try:
    cuts = config['vcuts']
    cut_method = 'VELOCITY'
except:
    pass
try:
    cuts = config['dcuts']
    cut_method = 'DISTANCE'
except:
    pass
try:
    cuts = config['cutfile']
    cut_method = 'FILE'
except:
    pass
if cut_method == 'NONE':
    print('ERROR: you need to provide cuts to separate components along the l.o.s')

# mandatory parameters
lmin = config['lmin']
lmax = config['lmax']
bmin = config['bmin']
bmax = config['bmax']
vmin = config['vmin']
vmax = config['vmax']
outdir = config['outdir']
names = list(config['names'].values())
if len(config['infile']) > 1:
    target_res = config['target_res']

#### initial map creation ####################################################################

if nthread > 1:
    pool = Pool(processes=min(len(config['infile']),nthread))

for s in range(len(config['infile'])):
    try:
        fitres_files = [config['fitres'][s], config['fitdiag'][s]]
    except:
        fitres_files = [None, None]
        useFit = False
    infile = config['infile'][s]
    try:
        rebin = config['rebin'].values()[s]
    except:
        rebin = False
    anomaly_file = outdir + '/{}anomalies.dat'.format(name_tag[s])
    args = (infile, Ts, fitres_files,
            lmin, lmax, bmin, bmax, vmin, vmax,
            names, cut_method, cuts, name_tag[s],
            useFit,outdir,
            T_anomaly, border, anomaly_file,inpaint)
    if nthread > 1:
        pool.apply_async(create_maps, args)
    else:
        create_maps(*args)

if nthread > 1:
    pool.close()
    pool.join()

##### merge maps if more than one input file ##################################################
if len(config['infile']) > 1:

    if nthread > 1:
        pool = Pool(processes=min(len(names),nthread))

    for name in names:
        if T_anomaly and inpaint:
            filenames = [outdir + 'lbmap_inpainted_' + name_tag[s] + name + '.fits'
                         for s in range(len(config['infile']))]
            # check that inpainted files exist
            new_filenames = []
            for fname in filenames:
                if os.path.exists(fname):
                    new_filenames.append(fname)
                else:
                    new_filenames.append(outdir +'lbmap' +fname.split('/')[-1][15:])
            filenames = new_filenames
        else:
            filenames = [outdir + 'lbmap' + name_tag[s] + name + '.fits'
                         for s in range(len(config['infile']))]
        mergedfile = outdir + 'lbmap_merged_' + name + '.fits'
        args = (filenames,mergedfile,
                target_res,
                lmin,lmax,bmin,bmax)
        if nthread > 1:
            pool.apply_async(merge_maps, args)
        else:
            merge_maps(*args)

    if nthread > 1:
        pool.close()
        pool.join()

