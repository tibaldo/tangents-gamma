# -*- coding: utf-8 -*-
from pylab import *
import sys, os, time
import numpy as np

#########################################################################################################################   
    
def Lagrangian5pts(T):
    #5-pt lagrangian derivative
    
    dim=len(T)
    #derivees
    dTdv = zeros(dim)
    for iv in range(2,dim-2): #5-pt lagrangian derivative
        dTdv[iv] = (T[iv-2] - 8.0*T[iv-1] + 8.0*T[iv+1] - T[iv+2]) / 12. 
    return dTdv

#########################################################################################################################   

def epsDetect(T,lng=3,sig=1,graphes=0,lis=0,name=time.strftime('%d-%m-%y_%H-%M',time.localtime())):
    now=name
    #adapte depuis code matlab de detection de raie HI     
    #indl=abs(T-moy)<1*rms
    dim=len(T)
    ## prepa gaussienne pour lissage
    sig =sig ; # bin
    dsig2 = 2. * sig * sig ;
    nbin=10
    x = arange(-nbin,nbin+1); # bins
    gauss = exp(- (x**2.) / dsig2) ;
    gauss = gauss / sum(gauss) ;
    
    Tlis= convolve(T,gauss,'same')
    #derivees 2nd des spectres
    if lis:
        dTdv=Lagrangian5pts(Tlis)
    else:
        dTdv=Lagrangian5pts(T)
        
    d2Tdv2=Lagrangian5pts(dTdv)
    #rms_d2Tdv2=std(d2Tdv2)
    cut=nbin
    rms_d2Tdv2=std(d2Tdv2[cut:-cut])
    
    #recherche des intervalles de longueur > lonth avec d2Tdv2 < 0 continument
    lonth =lng #nb mini de bins contigus avec d2Tdv2 < 0
    Nband = 0 
    indth = find(d2Tdv2 < 0.)
    nbth = len(indth) 
    iband1 = zeros(nbth) 
    iband2 = zeros(nbth) 
    ii1 = 0 
    while ii1 < nbth-2:
        ii2 = ii1 + 1 
        while (ii2 < nbth-1) & (indth[ii2] - indth[ii1] == ii2-ii1):
            ii2 = ii2 + 1 
        
        if ii2-ii1 >= lonth: #((ii2-1) - ii1 + 1)
            Nband = Nband + 1 
            iband1[Nband] = indth[ii1]
            iband2[Nband] = indth[ii2-1] 
        
        ii1 = ii2 
    
    ind = logical_not(iband1 == 0)
    iband1=iband1[ind]
    ind = logical_not(iband2 == 0) 
    iband2=iband2[ind]
    iband1 = iband1.astype('int')
    iband2 = iband2.astype('int')
    
    #recherche des minima de d2Tdv2 dans les intervalles negatifs
    ilin = []
    eps =[] 
    if Nband > 0:
        for iba in range(Nband):
            test = d2Tdv2[iband1[iba]:iband2[iba]] 
            imin = argmin(test)  #min sur bande d2Tdv2 < 0 concernee
            iv_imin = iband1[iba] + imin  #indicage dans [0,dimv-1]
            if np.any(T[iband1[iba]:iband2[iba]]):#change to eliminate zero-peaks
                ilin.append(iv_imin)
            #if min(d2Tdv2[iband1[iba]:iband2[iba]]) < -rms_d2Tdv2:
            #    ilin.append(iv_imin) 
                            
        ilin = sort(ilin) 
        Nlin = len(ilin)
        for ii in range(Nlin):
            eps.append(T[ilin[ii]])
    
    if graphes:
        figure(7)
        hold(True)
        figure(7)
        plot(range(dim-2*cut),d2Tdv2[cut:-cut], 'b-')
        hlines(rms_d2Tdv2,[0],dim,linestyles='dashed',colors='k')
        hlines(-rms_d2Tdv2,[0],dim,linestyles='dashed',colors='k')
        vlines(ilin,[0],d2Tdv2[ilin.astype(int)],linestyles='solid',colors='k')
        vlines(ilin,[0],-d2Tdv2[ilin.astype(int)],linestyles='solid',colors='k')        
                        
                                            
        plt.figure(facecolor='white',figsize=(15,9), dpi=100)
        plt.hold(True)
        plot(range(dim),T, 'k-')
        plot(range(dim),Tlis, 'g-')   
        #plt.hlines(eps,[0],ilin,linestyles='dashed',colors='r')
        plt.vlines(ilin,[0],eps,linestyles='solid',colors='r')   
        plt.tight_layout()
        plt.savefig("/Users/Quentin/Work/ISMsci/LineDetect15/line_detect_"+now+".png", dpi=plt.gcf().dpi) #, bbox_inches='tight'   
        
    return ilin, eps
        
