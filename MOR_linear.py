# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import utility as ut

import numpy as np
import numpy.linalg as nlina
import scipy.linalg as slina
import scipy.sparse.linalg as sslina
import oct2py
import os
import timeit

import port
import plotfuns
import MOR

oc = oct2py.Oct2Py()
pRead=lambda path:ut.pRead(oc, path)

path=dict()
path['workDir'] = os.path.abspath('').split('\\pythonTests')[0]+'\\pythonTests\\'
path['mats']    = path['workDir']+'mats\\'
path['A']       = path['mats']+'A\\'
path['ports']   = path['mats']+'ports\\'
path['RHSOff']  = path['mats']+'RHSOff\\'
path['RHSOn']   = path['mats']+'RHSOn\\'
path['JSrcOn']  = path['mats']+'JSrcOn\\'
path['plots']   = path['workDir']+'_Documentation\\images\\'


# nBasiss=np.linspace(1,30,30).astype(int)
nBasiss=[1,2,4,8,15,20,25,30,35,40,45,50,55,60]
# nBasiss=[1,15,30]
# nBasiss=[30]

nPorts=2
nModes={'TB':0,  ##Ist das korrekt?
        'TS':0,
        'TE':0,
        'TM':10}

cond=5.8e6

#==============================================================================
#import constant matrices
#==============================================================================

ports=[]
for i in range(nPorts):
    ports.append(port.port (path['ports']+str(i)+'\\',pRead,nModes))
    ports[i].readModes()
    ports[i].readMaps()


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



#==============================================================================
#precomputations:
#==============================================================================
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
res_curves=np.zeros((len(nBasiss),len(fAxisOn)))
err_curves=np.zeros((len(nBasiss),len(fAxisOn)))
Z=np.zeros((len(nBasiss),len(fAxisOn))).astype('complex')
res_Plot=np.zeros(len(nBasiss))
err_Plot=np.zeros(len(nBasiss))

resDotsX=[]
resDotsY=[]

for iBasis in range(len(nBasiss)):
    #==============================================================================
    #offline stage:
    #read data according to given points (done)
    #create basis matrix (done)
    #exploit affine decomposition  (todo)
    #select initial set of points (done)

    nBasis=nBasiss[iBasis]
    print('nBasis: '+str(nBasis))
    offInds=np.round(np.linspace(0, NOff-1, nBasis)).astype(int)
    # offInds=np.round(np.linspace(0, NOff-1, nBasis+2)).astype(int)[1:-1]
    SolsOffSelected=SolsOff[:,offInds]
    # SolsOffSelected/=nlina.norm(SolsOffSelected,axis=0)
    (U,s,V)=slina.svd(SolsOffSelected,full_matrices=False)

    #==============================================================================
    #online stage:
    #assemble matrix
    #calculate solution in each online point (todo)
    #calculate solution error (done)
    #calculate RHS residual   (todo)
    #select point(s) with highest residual (todo)

    (resTot,res_ROM,err_R_F,Znew,uROM)=MOR.podOnlineSt2(fAxisOn,U,ports,Mats,factors,RHSOn,SolsOn,JSrcOn)

    res_Plot[iBasis]   = resTot
    res_curves[iBasis] = res_ROM
    err_Plot[iBasis]   = nlina.norm(err_R_F)
    err_curves[iBasis] = err_R_F
    Z[iBasis]          = Znew


    (onFreqs,onInds)=ut.closest_values(fAxisOn, fAxisOff[offInds], returnInd=True)
    resDotsX.append(onFreqs)
    resDotsY.append(res_ROM[onInds])

    # print('res=' +str(resTot)+', singular vals:')
    # print(np.abs(s))
    ###############################################################################
    #post processing
    #calculate impedance

Zlin=Znew

ZRef=np.zeros(len(fAxisOn)).astype('complex')
# errPP1=np.zeros(len(fAxisOn))
# errPP2=np.zeros(len(fAxisOn))
for i in range(len(fAxisOn)):
    ZRef[i]=MOR.impedance(SolsOn[:,i], JSrcOn[:,i])
    # errPP1[i]=nlina.norm(SolsOff[:,i]-uROM[i])
    # errPP2[i]=nlina.norm((np.abs(JSrcOn[:,0])>0).astype('int')*(SolsOff[:,i]-uROM[i]))


# fig=plotfuns.initPlot(title='err,res',logX=True,logY=True,xName='f',yName='val')
# plotfuns.plotLine (fig, fAxisOn, err_R_F,lineArgs={'name':'err','color':0})
# plotfuns.plotLine (fig, fAxisOn, res_ROM,lineArgs={'name':'res','color':1})
# plotfuns.showPlot(fig)

fig=plotfuns.initPlot(title='res convergence',logX=True,logY=True,xName='nBasis',yName='val')
plotfuns.plotLine (fig, nBasiss, res_Plot,lineArgs={'name':'res','color':0})
plotfuns.plotLine (fig, nBasiss, err_Plot,lineArgs={'name':'err','color':1})
plotfuns.showPlot(fig)
# plotfuns.exportPlot(fig, 'const_conv_st2_broad1', 'full', path=path['plots'],opts={'legendShow':True})


# raise Exception()
# inds=np.unique((ut.logspace(1, 51, 10)-1).astype(int))
# inds=np.array([1,2,6,7,8,10,12,14,20,29])
# inds=np.linspace(1,len(nBasiss)-1,len(nBasiss)-1).astype(int)
inds=[0,3,6,9,13]
indsNames=[str(ind+1) for ind in inds]
selectedXDots=[resDotsX[i] for i in inds]
selectedYDots=[resDotsY[i] for i in inds]

fig=plotfuns.initPlot(title='res over freq',logX=False,logY=True,xName='f',yName='val')
fig.add_vline(x=fAxisOn[123], line_width=3, line_dash="dash", line_color="grey")
fig.add_vline(x=fAxisOn[188], line_width=3, line_dash="dash", line_color="grey")
plotfuns.plotLines (fig, fAxisOn, res_curves[inds,:],curveName=indsNames)
plotfuns.plotClouds (fig, selectedXDots, selectedYDots,cloudName=indsNames,showLegend=False)
# plotfuns.plotLines (fig, fAxisOn, err_curves)
plotfuns.showPlot(fig)
# plotfuns.exportPlot(fig, 'sqcR_conv_st2_broad2', 'full', path=path['plots'],opts={'legendShow':True})



raise Exception()

inds=[1]
indsNames=[str(ind+1) for ind in inds]

fig=plotfuns.initPlot(title='RE',logX=False,logY=False,xName='f',yName='Ohm')
plotfuns.plotLines (fig, fAxisOn, np.real(Z)[inds,:],curveName=indsNames)
plotfuns.showPlot(fig)
# plotfuns.exportPlot(fig, 'sqc_imp_MOR_RE', 'full', path=path['plots'],opts={'legendShow':True,'yRange':(-11e3,19e3)})

fig=plotfuns.initPlot(title='IM',logX=False,logY=False,xName='f',yName='Ohm')
plotfuns.plotLines (fig, fAxisOn, np.imag(Z)[inds,:],curveName=indsNames)
plotfuns.showPlot(fig)
# plotfuns.exportPlot(fig, 'sqc_imp_MOR_IM', 'full', path=path['plots'],opts={'legendShow':True,'yRange':(-5e3,70e3)})

fig=plotfuns.initPlot(title='abs',logX=False,logY=False,xName='f',yName='Ohm')
plotfuns.plotLines (fig, fAxisOn, np.abs(Z)[inds,:],curveName=indsNames)
plotfuns.showPlot(fig)
# plotfuns.exportPlot(fig, 'sqc_imp_MOR_IM', 'full', path=path['plots'],opts={'legendShow':True,'yRange':(-5e3,70e3)})

fig=plotfuns.initPlot(title='abs',logX=False,logY=True,xName='f',yName='Ohm')
plotfuns.plotLine (fig, fAxisOn, np.abs(Z[1,:])  ,lineArgs={'name':'MOR_2','color':-1} )
plotfuns.plotLine (fig, fAxisOn, np.abs(Z[2,:])  ,lineArgs={'name':'MOR_4','color':0} )
plotfuns.plotLine (fig, fAxisOn, np.abs(Z[3,:])  ,lineArgs={'name':'MOR_8','color':2} )
# plotfuns.plotLine (fig, fAxisOn, np.abs(Z[5,:])  ,lineArgs={'name':'MOR_20','color':1} )
plotfuns.plotLine (fig, fAxisOn, np.abs(ZRef) ,lineArgs={'name':'FEL','color':3} )
plotfuns.showPlot(fig)
# plotfuns.exportPlot(fig, 'const_imp_Felis_Py2', 'full', path=path['plots'],opts={'legendShow':True})
print('')




fig=plotfuns.initPlot(title='err/res over freq',logX=False,logY=True,xName='f',yName='val')
plotfuns.plotLines (fig, fAxisOn, [res_ROM],curveName='res')
plotfuns.plotLines (fig, fAxisOn, [err_R_F],curveName='err_R_F')
plotfuns.plotLines (fig, fAxisOn, [err_P_F],curveName='err_P_F')
plotfuns.plotClouds (fig, selectedXDots, selectedYDots,cloudName=indsNames,showLegend=False)
# plotfuns.plotLines (fig, fAxisOn, err_curves)
plotfuns.showPlot(fig)


#
# def obj_f(X):
#     x=X[0,:]
#     y=X[1,:]
#     res=np.zeors(len(x))
#     for i in len(x):
#         res[i]=x[i]**2 * y[i]
#     return res






