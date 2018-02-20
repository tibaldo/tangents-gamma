# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 00:03:12 2017

@author: Quentin
"""
import numpy
from numba import jit

def pseudoVoigt_mini(params,xin,nblines,T):
    # nb of gaussian lines with free width; no flat level
    # Hauteurs = param(1:nblines) ;
    # avg = param(nblines+1:2*nblines) ;
    # width = param(2*nblines+1:3*nblines) ;
    # eta = param(3*nblines+1:4*nblines) ;
	
    fitparam=numpy.zeros(4*nblines)
    for k in range(4*nblines):
        name='param'+str(k)
        fitparam[k]=params[name].value 
	
    nblin2 = 2 * nblines
    nblin3 = 3 * nblines 	
    output = 0. 

    for ii in range(nblines):
        aux=((xin-fitparam[ii+nblines])/ fitparam[ii+nblin2])**2.
        G=fitparam[ii] * numpy.exp(-0.5*aux)
        L=fitparam[ii] /(1.+1.*aux)
        PV=fitparam[ii+nblin3]*L+(1.-fitparam[ii+nblin3])*G
        output = output + PV
    return output-T

@jit
def pseudoVoigt(hfit,vfit,svfit,etafit,xin):
    aux=((xin-vfit)/ svfit)**2.
    G=hfit * numpy.exp(-0.5*aux)
    L=hfit /(1.+1.*aux)
    PV=etafit*L+(1.-etafit)*G
    return PV