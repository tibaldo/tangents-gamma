from MW_utils import *
from reid19_rotation import rotcurve, R0, V0
from reid19_arms import arm_polar, Narm

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from astropy.io import fits

# helper class to draw lines clicking on figures
class LineDrawer(object):
    lines = []
    def draw_line(self):
        ax = plt.gca()
        xy = plt.ginput(2)

        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        line = plt.plot(x,y,color='k',linestyle=':')
        ax.figure.canvas.draw()

        self.lines.append(line)

# helper function to get lon-dist curves from lon-vlsr boundaries
# through piece-wise linear interpolation
def ld_linear(l,v):
    # longitude values
    lon = np.linspace(np.min(l),np.max(l),100)
    vlsr = np.interp(lon,l,v)
    rad, dist = lbvlsr2rd0(lon,0,vlsr,R0,V0,rotcurve)
    return lon, dist


comap = fits.open("/Users/ltibaldo/Fermi/ISM/CO/COGAL_deep_mom.fits")[0].data
himap = fits.open("/Users/ltibaldo/Fermi/ISM/HI/HI4PI/CAR_E03.fits")[0].data
dustmap = fits.getdata('/Users/ltibaldo/Fermi/ISM/dust/extinction_cubes/machete_june_2019.fits',0)
dustmap2 = fits.getdata('/Users/ltibaldo/Fermi/ISM/dust/extinction_cubes/stilism2019_lbdcube_firstquad_compressed.fits',1)

#### read bessel SFR
filename = '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/parallax_data.inp'
names=['source','arm','glon','glat','vlsr','unc_vlsr','parallax','unc_parallax']
dtypes=['S12','S3','f8','f8','f8','f8','f8','f8']
bessel = np.genfromtxt(filename,skip_header=1,names=names,dtype=dtypes)

####color coding#########################################
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
labels = ['3 kpc', 'Norma/Outer', 'Scutum/Centaurus/OSC',
          'Sagittarius/Carina', 'Local', 'Perseus']
colors = [color_cycle[9],color_cycle[1],color_cycle[2],
          color_cycle[3],color_cycle[0],color_cycle[6]]
col=[colors[0],colors[0],
     colors[1],colors[1],colors[1],colors[1],colors[1],
     colors[2],colors[2],colors[2],colors[2],colors[2],
     colors[3], colors[3], colors[3],
     colors[4],
     colors[5]]
lab = ['3 kpc','Norma/outer','Sagittarius/Carina','Perseus','Scutum/Centaurus/OSC', 'local']
coldic = {'3kN' : colors[0], '3kF' : colors[0],
          'Nor' : colors[1], 'Out' : colors[1], 'OuX' : colors[1],
          'ScN' : colors[2], 'ScF' : colors[2], 'OSC' : colors[2], 'CtN' : colors[2], 'ScS' : colors[2],
          'SgN' : colors[3], 'SgF' : colors[3], 'Sgr' : colors[3], 'CrN' : colors[3],
          'Loc' : colors[4],
          'Per' : colors[5],
          'GC' : '0.3', 'Con' : '0.3', '???' : '0.3', 'AqS' : '0.3',
          'LoS' : '0.3'}

#########################################################
# Plots
plt.ion()

# outside view
fig0 = plt.figure("outsideview", figsize=(6, 5.5))
fig0.subplots_adjust(left=0.14,right=0.95, top=0.95)
ax0 = plt.subplot(111)
ax0.set_xlabel("[kpc]", fontsize=14)
ax0.set_ylabel("[kpc]", fontsize=14)
for tick in ax0.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax0.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')

ax0.scatter([0], [0], marker = 'x', s=50,facecolor='black', edgecolors='black',zorder=3)
ax0.plot([-15, 15], [R0, R0], color='black', linewidth=0.5)
ax0.plot([0, 0], [-18, 18], color='black', linewidth=0.5)
ax0.text(8, R0+0.3, "$l=90^\circ$", usetex=True)
ax0.text(-9.5, R0+0.3, "$l=-90^\circ$", usetex=True)
ax0.text(-1.75, -4.5, "$l=0^\circ$", usetex=True)
ax0.text(-2.25, 14., "$l=180^\circ$", usetex=True)
ax0.set_xlim(-10.,10.)
ax0.set_ylim(-5.,15.)
ax0.set_aspect('equal')

# overlay spiral arm model
for s in range(Narm):
    x, y = arm_xy(s,arm_polar)
    ax0.plot(x, y,linewidth=3,alpha=0.5,color=col[s])

#CO
fig1 = plt.figure("lonvel", figsize=(12, 4))
fig1.subplots_adjust(left=0.08, right=0.97, top=0.99, bottom=0.14)
ax1 = plt.subplot(111)
ax1.set_xlabel("Galactic longitude [$^\circ$]", fontsize=14, usetex=True)
ax1.set_ylabel("velocity [km s$^{-1}$]", fontsize=14, usetex=True)
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax1.xaxis.set_ticks_position('both')
ax1.yaxis.set_ticks_position('both')

comap[np.isnan(comap) == True] = 0.
comap = np.sum(comap, axis=0)
comap = comap.T
comap[comap < 0.1] = 0.1
co_plot = ax1.imshow(np.sqrt(comap[125:-126, :]), cmap="Spectral_r", origin='lower', extent=[180, -180, -157.3, 157.3])
ax1.set_aspect("auto", adjustable="box")
ax1.xaxis.set_ticks(np.arange(180, -210, -30))

#HI
fig2 = plt.figure("lonvelhi", figsize=(6, 5))
fig2.subplots_adjust(left=0.16, right=0.97, top=0.99, bottom=0.14)
ax2 = plt.subplot(111)
ax2.set_xlabel("Galactic longitude [$^\circ$]", fontsize=14, usetex=True)
ax2.set_ylabel("velocity [km s$^{-1}$]", fontsize=14, usetex=True)
for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax2.xaxis.set_ticks_position('both')
ax2.yaxis.set_ticks_position('both')

himap[np.isnan(himap) == True] = 0.
himap = np.sum(himap[:,72:193,:], axis=1)
himap[himap < 0.1] = 0.1
hi_plot = ax2.imshow(np.sqrt(himap[347:585, :]), cmap="Spectral_r", origin='lower', extent=[61.083, 39., -153.17, 153.17])
ax2.set_aspect("auto", adjustable="box")

#dust extinction
fig3 = plt.figure("londist", figsize=(10, 4))
fig3.subplots_adjust(left=0.08, right=0.97, top=0.99, bottom=0.14)
ax3 = plt.subplot(111)
ax3.set_xlabel("Galactic longitude [$^\circ$]", fontsize=14, usetex=True)
ax3.set_ylabel("distance [kpc]", fontsize=14, usetex=True)
for tick in ax3.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax3.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax3.xaxis.set_ticks_position('both')
ax3.yaxis.set_ticks_position('both')

dustmap[np.isnan(dustmap) == True] = 0.
dustmap = np.sum(dustmap,axis=1)#integrate latitude
dustmap[dustmap<0]=0

dust_plot = ax3.imshow(np.log(dustmap), origin='lower', extent=[180, -180, 0., 20.],vmin=1,vmax=5,cmap='Spectral_r')
ax3.set_aspect("auto", adjustable="box")
ax3.xaxis.set_ticks(np.arange(180, -180, -10))

#dust extinction 2
fig4 = plt.figure("londist2", figsize=(8, 4))
fig4.subplots_adjust(left=0.08, right=0.97, top=0.99, bottom=0.14)
ax4 = plt.subplot(111)
ax4.set_xlabel("Galactic longitude [$^\circ$]", fontsize=14, usetex=True)
ax4.set_ylabel("distance [kpc]", fontsize=14, usetex=True)
for tick in ax4.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax4.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax4.xaxis.set_ticks_position('both')
ax4.yaxis.set_ticks_position('both')

dustmap2[np.isnan(dustmap2) == True] = 0.
dustmap2 = np.sum(dustmap2[:,260:341,:],axis=1)#integrate latitude
dustmap2[dustmap2<0]=0

dust_plot2 = ax4.imshow(np.log(dustmap2), origin='lower', extent=[90, 0, 0., 3.],cmap='Spectral_r')
ax4.set_aspect("auto", adjustable="box")
ax4.xaxis.set_ticks(np.arange(90, 0, -10))

# # overlay spiral arm model
for s in range(Narm):
    ll, vlsr = arm_lv(s,R0, V0, arm_polar, rotcurve)
    ll, dd = arm_ld0(s, R0, arm_polar)
    if s == Narm -1: # Perseus
        ax1.plot(ll[ll>-30],vlsr[ll>-30],color=col[s],linewidth=3,alpha=0.3)
        ax1.plot(ll[ll < -30], vlsr[ll < -30], color=col[s], linewidth=3, alpha=0.3)
        ax2.plot(ll[ll>-30], vlsr[ll>-30], color=col[s], linewidth=3, alpha=0.3)
        ax2.plot(ll[ll < -30], vlsr[ll < -30], color=col[s], linewidth=3, alpha=0.3)
        ax3.plot(ll[ll>-30], dd[ll>-30], color=col[s], linewidth=3, alpha=0.3)
        ax3.plot(ll[ll < -30], dd[ll < -30], color=col[s], linewidth=3, alpha=0.3)
        ax4.plot(ll[ll>-30], dd[ll>-30], color=col[s], linewidth=3, alpha=0.3)
        ax4.plot(ll[ll < -30], dd[ll < -30], color=col[s], linewidth=3, alpha=0.3)
    else:
        ax1.plot(ll[ll>0],vlsr[ll>0],color=col[s],linewidth=3,alpha=0.3)
        ax1.plot(ll[ll < 0], vlsr[ll < 0], color=col[s], linewidth=3, alpha=0.3)
        ax2.plot(ll[ll>0], vlsr[ll>0], color=col[s], linewidth=3, alpha=0.3)
        ax2.plot(ll[ll < 0], vlsr[ll < 0], color=col[s], linewidth=3, alpha=0.3)
        ax3.plot(ll[ll>0], dd[ll>0], color=col[s], linewidth=3, alpha=0.3)
        ax3.plot(ll[ll < 0], dd[ll < 0], color=col[s], linewidth=3, alpha=0.3)
        ax4.plot(ll[ll>0], dd[ll>0], color=col[s], linewidth=3, alpha=0.3)
        ax4.plot(ll[ll < 0], dd[ll < 0], color=col[s], linewidth=3, alpha=0.3)

#overlay BeSSel high-mass SFR
for obj in bessel:
    dist = 1./obj['parallax']#distance from the Earth in kpc
    l = obj['glon']
    if l>180:
        l -=360.

    # marker size inversely proportional to vlsr unc
    if obj['unc_vlsr'] <= 3.:
        size = 6.
    elif obj['unc_vlsr'] < 7:
        size = 3.
    else:
        size = 1.5
    ax1.plot(l, obj['vlsr'], marker='o', markersize=size, color=coldic[obj["arm"]], alpha=0.7)
    ax2.plot(l, obj['vlsr'], marker='o', markersize=size, color=coldic[obj["arm"]], alpha=0.7)

    # marker size inversely proportional to distance unc
    unc_dist = dist * obj['unc_parallax'] / obj['parallax']
    if unc_dist < 0.5:
        size = 6
    elif unc_dist < 1.:
        size = 3
    else:
        size = 1.5
    ax3.plot(l, dist, marker='o', markersize=size, color=coldic[obj["arm"]], alpha=0.7)
    ax4.plot(l, dist, marker='o', markersize=size, color=coldic[obj["arm"]], alpha=0.7)
    x, y = lbd2xy(l,obj['glat'],dist,R0)
    ax0.plot(x, y, marker='o', markersize=size, color=coldic[obj["arm"]], alpha=0.7)

# make custom legend
legend_elements = []
for s, label in enumerate(labels):
    el = Line2D([0], [0], color=colors[s], lw=2, label=label)
    legend_elements.append(el)

ax1.legend(handles=legend_elements)
ax3.legend(handles=legend_elements)
#ax4.legend(handles=legend_elements)
