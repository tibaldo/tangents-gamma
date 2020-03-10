import sys
sys.path.append('../ISM/')

from MW_utils import *
from reid14_rotation import rotcurve, R0, V0
from ne2001_arms import arm_polar, Narm

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib.patches import Circle

comap = fits.open("/Users/ltibaldo/Fermi/ISM/CO/COGAL_deep_mom.fits")[0].data # Change to the folder containing the files
himaps = fits.open("/Users/ltibaldo/Fermi/ISM/HI/HI4PI/CAR_E02.fits")[0].data
bessel = fits.getdata("/Users/ltibaldo/Fermi/tangents/images2/asu.fit", 1)
dustmap = fits.getdata('/Users/ltibaldo/Fermi/ISM/dust/extinction_cubes/machete_june_2019.fits',0)

####color coding#########################################
col=["#4C72B0", "#55A868", "#C44E52","#8172B2", "#CCB974"]
lab = ['Norma/outer','Sagittarius/Carina','Perseus','Scutum/Centaurus', 'local spur']
coldic = {"Out": 'b', "Sgr":'g',"Per":'r',"Loc":'orange',"Sct":'indigo'}

#########################################################

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

ax0.scatter([0], [0], facecolor='white')
ax0.scatter([0], [8.29], facecolor='None', edgecolors='black')
ax0.plot([-15, 15], [8.29, 8.29], color='black', linewidth=0.5)
ax0.plot([0, 0], [-18, 18], color='black', linewidth=0.5)
ax0.text(16, 7.9, "$l=90^\circ$", usetex=True)
ax0.text(-19.5, 7.9, "$l=-90^\circ$", usetex=True)
ax0.text(-1.25, -19, "$l=0^\circ$", usetex=True)
ax0.text(-1.45, 18.5, "$l=180^\circ$", usetex=True)
ax0.set_xlim(-20.,20.)
ax0.set_ylim(-20.,20.)

# overlay spiral arm model
for s in range(Narm):
    x, y = arm_xy(s,arm_polar)
    ax0.plot(x, y,linewidth=10,alpha=0.5,color=col[s])

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
co_plot = ax1.imshow(np.sqrt(comap[125:-126, :]), cmap="gray_r", origin='lower', extent=[180, -180, -157.3, 157.3])
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


# Val = CRVAL + (PIX - CRPIX)*CDELT
#himaps
himaps[np.isnan(himaps) == True] = 0. #Fortan pixels start at 1, Python pixels start at 0
himaps = np.sum(himaps[:,72:193,:], axis=1) # latitude integral between [-5,+5] degrees
himaps[himaps < 0.1] = 0.1 # lowers background noise
hi_plot = ax2.imshow(np.sqrt(himaps[347:585, :]), cmap="gray_r", origin='lower', extent=[41.08, 19., -153.17, 153.17]) # plot montre valeur image longitude vitesse avec limitation en vitesse
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
# for s in range(len(dustmap)-1,0,-1):
#     dustmap[s] = dustmap[s] - dustmap[s-1]
dustmap = np.sum(dustmap,axis=1)#integrate latitude
dustmap[dustmap<0]=0

dust_plot = plt.imshow(np.log(dustmap), origin='lower', extent=[180, -180, 0., 20.],vmin=1,vmax=5,cmap='gray_r')
ax3.set_aspect("auto", adjustable="box")
ax3.xaxis.set_ticks(np.arange(180, -180, -10))


# overlay spiral arm model
for s in range(Narm):
    ll, vlsr = arm_lv(s,R0, V0, arm_polar, rotcurve)
    ll, dd = arm_ld0(s, R0, arm_polar)
    if s==2:
        ax1.plot(ll[ll>0],vlsr[ll>0],linewidth=10,color=col[s],alpha=0.5,label=lab[s])
        ax1.plot(ll[(ll<0) & (ll>-14)],vlsr[(ll<0) & (ll>-14)],linewidth=10,color=col[s],alpha=0.5)
        ax1.plot(ll[ll<-14],vlsr[ll<-14],linewidth=10,color=col[s],alpha=0.5)
        ax2.plot(ll[ll > 0], vlsr[ll > 0], linewidth=10, color=col[s], alpha=0.5, label=lab[s])
        ax2.plot(ll[(ll < 0) & (ll > -14)], vlsr[(ll < 0) & (ll > -14)], linewidth=10, color=col[s], alpha=0.5)
        ax2.plot(ll[ll < -14], vlsr[ll < -14], linewidth=10, color=col[s], alpha=0.5)
        ax3.plot(ll[ll > 0], dd[ll > 0], linewidth=10, color=col[s], alpha=0.5, label=lab[s])
        ax3.plot(ll[(ll < 0) & (ll > -14)], dd[(ll < 0) & (ll > -14)], linewidth=10,
                 color=col[s], alpha=0.5)
        ax3.plot(ll[ll < -14], dd[ll < -14], linewidth=10, color=col[s], alpha=0.5)
    elif s==0:
        ax1.plot(ll[ll>0],vlsr[ll>0],linewidth=10,color=col[s],label=lab[s],alpha=0.5)
        ax1.plot(ll[(ll<0) & (ll>-33)],vlsr[(ll<0) & (ll>-33)],linewidth=10,color=col[s],alpha=0.5)
        ax1.plot(ll[ll<-33],vlsr[ll<-33],linewidth=10,color=col[s],alpha=0.5)
        ax2.plot(ll[ll > 0], vlsr[ll > 0], linewidth=10, color=col[s], label=lab[s], alpha=0.5)
        ax2.plot(ll[(ll < 0) & (ll > -33)], vlsr[(ll < 0) & (ll > -33)], linewidth=10, color=col[s], alpha=0.5)
        ax2.plot(ll[ll < -33], vlsr[ll < -33], linewidth=10, color=col[s], alpha=0.5)
        ax3.plot(ll[ll > 0], dd[ll > 0], linewidth=10, color=col[s], label=lab[s], alpha=0.5)
        ax3.plot(ll[(ll < 0) & (ll > -33)], dd[(ll < 0) & (ll > -33)], linewidth=10,
                 color=col[s], alpha=0.5)
        ax3.plot(ll[ll < -33], dd[ll < -33], linewidth=10, color=col[s], alpha=0.5)
    elif s==4:
        ax1.plot(ll[ll>0],vlsr[ll>0],color=col[s],linewidth=10,label=lab[s],alpha=0.5)
        ax1.plot(ll[ll<0],vlsr[ll<0],color=col[s],linewidth=10,alpha=0.5)
        ax2.plot(ll[ll > 0], vlsr[ll > 0], color=col[s], linewidth=10, label=lab[s], alpha=0.5)
        ax2.plot(ll[ll < 0], vlsr[ll < 0], color=col[s], linewidth=10, alpha=0.5)
        ax3.plot(ll[ll > 0], dd[ll > 0], color=col[s], linewidth=10, label=lab[s], alpha=0.5)
        ax3.plot(ll[ll < 0], dd[ll < 0], color=col[s], linewidth=10, alpha=0.5)
    else:
        ax1.plot(ll,vlsr,color=col[s],linewidth=10,label=lab[s],alpha=0.5)
        ax2.plot(ll, vlsr, color=col[s], linewidth=10, label=lab[s], alpha=0.5)
        ax3.plot(ll, dd, color=col[s], linewidth=10, label=lab[s], alpha=0.5)
ax1.legend()
ax3.legend()


#overlay BeSSel high-mass SFR
for obj in bessel:
    ra = obj['RAJ2000']
    dec = obj['DEJ2000']
    vlsr = obj['VLSR']
    c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs') # c = SkyCoord('00 42 30 +41 12 00', unit=(u.hourangle, u.deg))
    l = c.galactic.l.deg
    b = c.galactic.b.deg
    dist = 1./obj['plx'] # distance from the Earth in kpc
    if l>180:
        l -=360.
    try:
        ax1.plot(l, vlsr, marker='^', color=coldic[obj["Arm"]], alpha=0.7)
        ax2.plot(l, vlsr, marker='^', color=coldic[obj["Arm"]], alpha=0.7)
        ax3.plot(l, dist, marker='^', color=coldic[obj["Arm"]], alpha=0.7)
    except:
        ax1.plot(l, vlsr, marker='^', color='k', alpha=0.3)
        ax2.plot(l, vlsr, marker='^', color='k', alpha=0.3)
        ax3.plot(l, dist, marker='^', color='k', alpha=0.3)
    x, y = lbd2xy(l,b,dist,R0) # Method LBD to XY L lOngi B lat D Distance = 1/parallax
    try:
        ax0.plot(x, y, marker='^', color=coldic[obj["Arm"]], alpha=0.7) # ignore that
    except:
        ax0.plot(x, y, marker='^', color='k', alpha=0.3) # Line to plot after getting lbd
    # if obj['Arm'] == 'Sgr' and l > 46.: # To ignore
    #     print('SFR')
    #     print(l,vlsr,dist)

#add boundaries for the study of Scutum tangent
bounds=[1.2,4.,9.39,12.4,15.3] # radius of Circles around the Sun 4.8 9.5
boundcolors = ['k','g','b','r','y'] # Color of Circles around the Sun

lmins = [26.,23.]
lmaxs = [35.,38.]
i=0
for i in [0,1]:
    lmin = lmins[i] # Longitude of line of sight no.1
    lmax = lmaxs[i] # Longitude of line of sight no.2
    x,y = lbd2xy(lmin,0,R0,R0) # Line of Sight delimiters
    y2 = R0 + 2*x*(y-R0)/x
    ax0.plot([0,2*x],[R0,y2],linestyle='--',color='k')
    x,y = lbd2xy(lmax,0,R0,R0)
    y2 = R0 + 2*x*(y-R0)/x
    ax0.plot([0,2*x],[R0,y2],linestyle='--',color='k')
    ax1.vlines([lmin,lmax],-157.3, 157.3,linestyle='--',color='k')
    ax2.vlines([lmin,lmax],-153.17, 153.17,linestyle='--',color='k')
    ax3.vlines([lmin,lmax],0, 15.,linestyle='--',color='k')

lvals=np.linspace(lmin,lmax,200)
for s, bound in enumerate(bounds): # Creates Circles around Sun
    circ = Circle((0,R0),bound,color=boundcolors[s], linewidth=1.5, linestyle='--',fill=False,zorder=3)
    ax0.add_patch(circ)
    vbound = lbd2vlsr(lvals,0,bound,R0,V0,rotcurve)
    dbound = bound * np.ones(len(lvals))
    ax1.plot(lvals,vbound,color=boundcolors[s], linewidth=1.5, linestyle='--')
    ax2.plot(lvals, vbound, color=boundcolors[s], linewidth=1.5, linestyle='--')
    ax3.plot(lvals, dbound, color=boundcolors[s], linewidth=1.5, linestyle='--')

plt.show()
