# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:39:17 2022

@author: Fred
"""

eps  = 8.8541878e-12 
mu = 1.2566371e-06 
c0=-1 
    

import utility as ut

import numpy as np
import numpy.linalg as nlina
import scipy.linalg as slina
import scipy.sparse.linalg as sslina
import oct2py
import os

import port
import plotfuns
import MOR

oc = oct2py.Oct2Py()
pRead=lambda path:ut.pRead(oc, path)



path=dict();
path['workDir'] = os.path.abspath('').split('\\pythonTests')[0]+'\\pythonTests\\'
path['mats']    = path['workDir']+'mats\\'
path['A']       = path['mats']+'A\\'
path['ports']   = path['mats']+'ports\\'
path['SIBc']    = path['mats']+'SIBc\\'
path['RHSOff']  = path['mats']+'RHSOff\\'
path['RHSOn']   = path['mats']+'RHSOn\\'
path['JSrcOn']  = path['mats']+'JSrcOn\\'
path['plots']   = path['workDir']+'_Documentation\\images\\'


nBasiss=np.linspace(1,50,50).astype(int)

nPorts=2
nModes={'TB':0,
        'TS':0,
        'TE':5,
        'TM':5,}
###############################################################################
#import constant matrices
fAxisOff = ut.csvRead (path['mats']+'fAxisOff.csv')[:,0]
fAxisOn  = ut.csvRead (path['mats']+'fAxisOn.csv' )[:,0]

CC      = pRead (path['mats']+'CC')

Sibc=[]
cond=5.8e3
norms=np.zeros(len(fAxisOff))
factors=np.zeros(len(fAxisOff)).astype('complex')
for i in range(len(fAxisOff)):
    Sibc.append (pRead (path['SIBc' ]+'SIBc' +str(i)))
    w=2*np.pi*fAxisOff[i]
    imp=np.sqrt(1j*mu*w/(cond+1j*eps*w))
    adm=1/imp
    factor=1j*w*mu*adm
    factors[i]=factor#cond+1j*eps*w
    # norms[i]=sslina.norm(Sibc[i],ord=np.inf)
    
    SibcAffine=Sibc[0]*factors[i]/factors[0]
    norms[i]=sslina.norm(Sibc[i]-SibcAffine)




# factorPlot=np.abs(factors/factors[0]*norms[0])
# mean=(factorPlot+norms)/2
# # factorPlot=np.abs(factors)
# inds=np.array([0,-1])
fig=plotfuns.initPlot(title='res convergence',logX=False,logY=False,xName='f',yName='val')
# plotfuns.plotLine (fig, fAxisOff, factorPlot,lineArgs={'name':'fact','color':0})
plotfuns.plotLine (fig, fAxisOff, norms,     lineArgs={'name':'norm','color':1})
# plotfuns.plotLine (fig, fAxisOff, mean,     lineArgs={'name':'mean','color':2})
# plotfuns.plotLine (fig, fAxisOff[inds], mean[inds],     lineArgs={'name':'mean','color':3})
plotfuns.showPlot(fig)

































