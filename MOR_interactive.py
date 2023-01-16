
import utility as ut


import numpy as np
import numpy.linalg as nlina
import scipy.linalg as slina
import scipy.sparse as ss
import scipy.sparse.linalg as sslina
import oct2py
import os
import zmq
import warnings

import port
import plotfuns
import MOR

oc = oct2py.Oct2Py()
pRead=lambda path:ut.pRead(oc, path)

path=dict();
path['workDir'] = os.path.abspath('').split('\\interactive_mor_felis')[0]+'\\interactive_mor_felis\\'
path['plots']   = path['workDir']+'_Documentation\\images\\'
path['mats']    = path['workDir']+'mats\\'
path['ports']   = path['mats']+'ports\\'
path['sols']    = path['mats']+'sols\\'
path['sysmats']    = path['mats']+'systemMats\\'


nMax =20
f_data= {'fmin' : 200e6,
         'fmax' : 1e9,
         'nmor' : 100,
         'ntest': 10,
         }

nPorts=2
nModes={'TB':0,
        'TS':0,
        'TE':0,
        'TM':10,}

cond=5.8e6

init_felis=True
recreate_mats=True
recreate_test=True

###############################################################################
###create constant matrices
(CC,ME,MC,Sibc,ports,RHS,JSrc,fAxis,fAxisTest,fIndsTest,sols_test,socket)=\
    MOR.createMatrices(init_felis,recreate_mats,recreate_test,pRead,path,cond,f_data['fmin'],f_data['fmax'],f_data['nmor'],f_data['ntest'],nPorts,nModes)


###############################################################################
###MOR Solving

(eps,mu,c0)=MOR.getFelisConstants()

w=lambda freq: 2*np.pi*freq
kap=lambda freq:w(freq)/c0
factorSibc= lambda freq:1j*w(freq)*mu/np.sqrt(1j*mu*w(freq)/(cond+1j*eps*w(freq)))

Mats=[CC,ME,MC,Sibc]
factors=[
    lambda f:1,
    lambda f:-kap(f)**2,
    lambda f:1j*w(f)*mu,
    lambda f:factorSibc(f)/factorSibc(fAxis[0])
        ]

res_Plot=[]
err_Plot=[]
res_curves=[]
resDotsX=[]
resDotsY=[]
Z=[]
nBasiss=[]
res_ROM=0
solInds=[]
for iBasis in range(nMax):
    ###############################################################################
    #offline stage:
    try:
        MOR.selectIndAdaptive(f_data['nmor'],res_ROM,solInds)
    except Exception:
        warnings.warn('No more frequencies to select')
        break

    socket.send_string("solve: train_%d  %f" %(iBasis,fAxis[solInds[-1]]))
    message = socket.recv()
    print(message)

    if iBasis == 0:
        sols= pRead(path['sols'] + 'train_0').toarray()
    else:
        sols = np.append(sols, pRead(path['sols'] +'train_'+str(iBasis)).toarray(), axis=1)

    (U,s,V)=slina.svd(sols,full_matrices=False)


    ###############################################################################
    #online stage:

    (resTot,res_ROM,err_R_F,Znew,uMOR)=MOR.podOnlineSt2(fAxis,U,ports,Mats,factors,RHS,fIndsTest,sols_test,JSrc)
    res_Plot.append(resTot)
    err_Plot.append(nlina.norm(err_R_F))
    res_curves.append(res_ROM)

    resDotsX.append(fAxis[solInds])
    resDotsY.append(res_ROM[solInds])
    Z.append(Znew)
    nBasiss.append(iBasis+1)

ZRef=np.zeros(len(fAxisTest)).astype('complex')
for i in range(len(fAxisTest)):
    ZRef[i]=MOR.impedance(sols_test[:,i], JSrc[:,fIndsTest[i]])


# fig=plotfuns.initPlot(title='err,res',logX=True,logY=True,xName='f',yName='val')
# plotfuns.plotLine (fig, fAxisOn, err_R_F,lineArgs={'name':'err','color':0})
# plotfuns.plotLine (fig, fAxisOn, res_ROM,lineArgs={'name':'res','color':1})
# plotfuns.showPlot(fig)

fig=plotfuns.initPlot(title='res convergence',logX=True,logY=True,xName='nBasis',yName='val')
plotfuns.plotLine (fig, nBasiss, res_Plot,lineArgs={'name':'res','color':0})
plotfuns.plotLine (fig, nBasiss, err_Plot,lineArgs={'name':'err','color':1})
plotfuns.showPlot(fig)
# plotfuns.exportPlot(fig, 'const_conv_st2_adap_1', 'full', path=path['plots'],opts={'legendShow':True})

raise Exception('stop')

fig=plotfuns.initPlot(title='abs',logX=False,logY=True,xName='f',yName='Ohm')
plotfuns.plotLine (fig, fAxis, np.abs(Z[-1])  ,lineArgs={'name':'MOR_ada','color':1} )
# plotfuns.plotLine (fig, fAxisOn, np.abs(ZLin)  ,lineArgs={'name':'MOR_lin','color':3} )
# plotfuns.plotLine (fig, fAxisOn, np.abs(Zlin)  ,lineArgs={'name':'MOR_lin','color':0} )
# plotfuns.plotLine (fig, fAxisOn, np.abs(Zada)  ,lineArgs={'name':'MOR_ada','color':1} )
plotfuns.plotLine (fig, fAxisTest, np.abs(ZRef) ,lineArgs={'name':'FEL','color':3} )
plotfuns.showPlot(fig)
# plotfuns.exportPlot(fig, 'sqcR_imp_ada1', 'full', path=path['plots'],opts={'legendShow':True})















