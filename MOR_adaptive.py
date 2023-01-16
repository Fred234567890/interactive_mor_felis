# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import utility as ut

import numpy as np
import numpy.linalg as nlina
import scipy.linalg as slina
import scipy.sparse.linalg as sslina
import oct2py
import os
import subprocess
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



nInit=3
nMax =10

nPorts=2
nModes={'TB':0,
        'TS':0,
        'TE':0,
        'TM':10,}

cond=5.8e3
###############################################################################
#import constant matrices
fAxisOff = ut.csvRead (path['mats']+'fAxisOff.csv')[:,0]
fAxisOn  = ut.csvRead (path['mats']+'fAxisOn.csv' )[:,0]

CC      = pRead (path['mats']+'CC'  )
MC      = pRead (path['mats']+'MC'  )
ME      = pRead (path['mats']+'ME'  )
try:
    Sibc    = pRead (path['mats']+'Sibc')
except:
    Sibc=ME*0
    
SolsOff = pRead (path['mats']+'solsOff').toarray()
SolsOn  = pRead (path['mats']+'solsOn' ).toarray()
RHSOn   = pRead (path['mats']+'RHSs'   ).toarray()
JSrcOn  = pRead (path['mats']+'Js'     ).toarray()

ports=[]
for i in range(nPorts):
    ports.append(port.port (path['ports']+str(i)+'\\',pRead,nModes))
    ports[i].readModes()
    ports[i].readMaps()

###############################################################################
#precomputations: 
NOff=len(fAxisOff)
    
eps  = 8.8541878e-12
mu = 1.2566371e-06
c0 = 1/np.sqrt(mu*eps)

w=lambda freq: 2*np.pi*freq

kap=lambda freq:w(freq)/c0

factorSibc= lambda f:1j*w(f)*mu/np.sqrt(1j*mu*w(f)/(cond+1j*eps*w(f)))

Mats=[CC,ME,MC,Sibc]
factors=[
    lambda f:1,
    lambda f:-kap(f)**2,
    lambda f:1j*w(f)*mu,
    lambda f:factorSibc(f)/factorSibc(fAxisOff[0])
        ]   
    
    
#MOR runs
res_ROM=0
nBasiss=[]
res_Plot=[]
err_Plot=[]
res_curves=[]
resDotsX=[]
resDotsY=[]
# Z=np.zeros((len(nBasiss),len(fAxisOn))).astype('complex')
Z=[]
for iBasis in range(nMax-nInit):
    ###############################################################################
    #offline stage: 
    #read data according to given points (done)
    #create basis matrix (done)
    #exploit affine decomposition  (todo)
    #select initial set of points (done)
    if iBasis==0:
        offInds=np.round(np.linspace(0, NOff-1, nInit+2)).astype(int)[1:-1]
    else:
        fAdd=fAxisOn [np.argmax(res_ROM)]
        (_,newInd)=ut.closest_value(fAxisOff, fAdd, returnInd=True)
        offInds=np.append(offInds,newInd)
        if not len(offInds)==len(np.unique(offInds)): raise Exception('selected point twice')
        
    print('nBasis: '+str(len(offInds)))
    nBasiss.append(len(offInds))
    
    SolsOffSelected=SolsOff[:,offInds]
    (U,s,V)=slina.svd(SolsOffSelected,full_matrices=False)



    ###############################################################################
    #online stage: 
    #assemble matrix
    #calculate solution in each online point (todo)
    #calculate solution error (done)
    #calculate RHS residual   (todo)
    #select point(s) with highest residual (todo) 
    
    (resTot,res_ROM,err_R_F,Znew,uMOR)=MOR.podOnlineSt2(fAxisOn,U,ports,Mats,factors,RHSOn,SolsOn,JSrcOn)
    res_Plot.append(resTot)
    err_Plot.append(nlina.norm(err_R_F))
    res_curves.append(res_ROM)
    
    (onFreqs,onInds)=ut.closest_values(fAxisOn, fAxisOff[offInds], returnInd=True)
    resDotsX.append(onFreqs)
    resDotsY.append(res_ROM[onInds])
    Z.append(Znew)
    ###############################################################################
    #post processing
    #calculate impedance
    
Zada=Znew
###############################################################################
#calculate reference impedance
ZRef=np.zeros(len(fAxisOn)).astype('complex')
for i in range(len(fAxisOn)):
    ZRef[i]=MOR.impedance(SolsOn[:,i], JSrcOn[:,i])



# fig=plotfuns.initPlot(title='err,res',logX=True,logY=True,xName='f',yName='val')
# plotfuns.plotLine (fig, fAxisOn, err_R_F,lineArgs={'name':'err','color':0})
# plotfuns.plotLine (fig, fAxisOn, res_ROM,lineArgs={'name':'res','color':1})
# plotfuns.showPlot(fig)

fig=plotfuns.initPlot(title='res convergence',logX=True,logY=True,xName='nBasis',yName='val')
plotfuns.plotLine (fig, nBasiss, res_Plot,lineArgs={'name':'res','color':0})
plotfuns.plotLine (fig, nBasiss, err_Plot,lineArgs={'name':'err','color':1})
plotfuns.showPlot(fig)
# plotfuns.exportPlot(fig, 'const_conv_st2_adap_1', 'full', path=path['plots'],opts={'legendShow':True})


raise Exception()
inds=np.unique((ut.logspace(3, 6, 5)-2).astype(int))
indsNames=[str(ind+1) for ind in inds]
selectedCurves=[res_curves[i] for i in inds]
selectedXDots=[resDotsX[i] for i in inds]
selectedYDots=[resDotsY[i] for i in inds]

fig=plotfuns.initPlot(title='res over freq',logX=False,logY=True,xName='f',yName='val')
fig.add_vline(x=fAxisOn[123], line_width=3, line_dash="dash", line_color="grey")
fig.add_vline(x=fAxisOn[188], line_width=3, line_dash="dash", line_color="grey")
plotfuns.plotLines (fig, fAxisOn, selectedCurves,curveName=indsNames)
# plotfuns.plotClouds (fig, selectedXDots, selectedYDots,cloudName=indsNames,showLegend=False)
# plotfuns.plotLines (fig, fAxisOn, err_curves)
plotfuns.showPlot(fig)
# plotfuns.exportPlot(fig, 'const_conv_st2_adap_1', 'full', path=path['plots'],opts={'legendShow':True})


inds=[0,2,4,6,8,18]
indsNames=[str(ind+1) for ind in inds]

fig=plotfuns.initPlot(title='RE',logX=False,logY=False,xName='f',yName='Ohm')
plotfuns.plotLines (fig, fAxisOn, np.real(Z)[inds,:],curveName=indsNames)
plotfuns.showPlot(fig)
# plotfuns.exportPlot(fig, 'sqc_imp_MOR_RE', 'full', path=path['plots'],opts={'legendShow':True,'yRange':(-11e3,19e3)})

fig=plotfuns.initPlot(title='IM',logX=False,logY=False,xName='f',yName='Ohm')
plotfuns.plotLines (fig, fAxisOn, -np.imag(Z)[inds,:],curveName=indsNames)
plotfuns.showPlot(fig)
# plotfuns.exportPlot(fig, 'sqc_imp_MOR_IM', 'full', path=path['plots'],opts={'legendShow':True,'yRange':(-5e3,70e3)})

print('')

# zRe=np.real(ut.polarCmplx(Znew))





fig=plotfuns.initPlot(title='RE',logX=False,logY=True,xName='f',yName='Ohm')
# plotfuns.plotLine (fig, fAxisOn, np.real(Znew),lineArgs={'name':'re','color':0})
# plotfuns.plotLine (fig, fAxisOn, np.imag(Znew),lineArgs={'name':'im','color':1})
plotfuns.plotLine (fig, fAxisOn, np.real(Znew) ,lineArgs={'name':'abs','color':3})
plotfuns.showPlot(fig)
# plotfuns.exportPlot(fig, 'sqcR_imp_ada1', 'full', path=path['plots'],opts={'legendShow':True})

fig=plotfuns.initPlot(title='IM',logX=False,logY=False,xName='f',yName='Ohm')
plotfuns.plotLine (fig, fAxisOn, -np.imag(Z)[inds,:],curveName=indsNames)
plotfuns.showPlot(fig)
# plotfuns.exportPlot(fig, 'sqc_imp_MOR_IM', 'full', path=path['plots'],opts={'legendShow':True,'yRange':(-5e3,70e3)})

fig=plotfuns.initPlot(title='abs',logX=False,logY=True,xName='f',yName='Ohm')
plotfuns.plotLine (fig, fAxisOn, np.abs(Znew)  ,lineArgs={'name':'MOR_ada','color':1} )
# plotfuns.plotLine (fig, fAxisOn, np.abs(ZLin)  ,lineArgs={'name':'MOR_lin','color':3} )
# plotfuns.plotLine (fig, fAxisOn, np.abs(Zlin)  ,lineArgs={'name':'MOR_lin','color':0} )
# plotfuns.plotLine (fig, fAxisOn, np.abs(Zada)  ,lineArgs={'name':'MOR_ada','color':1} )
# plotfuns.plotLine (fig, fAxisOn, np.abs(ZRef) ,lineArgs={'name':'FEL','color':3} )
plotfuns.showPlot(fig)
# plotfuns.exportPlot(fig, 'sqcR_imp_ada1', 'full', path=path['plots'],opts={'legendShow':True})























