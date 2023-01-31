
import utility as ut


import numpy as np
import numpy.linalg as nlina
import scipy.linalg as slina
import scipy.sparse as ss
import scipy.sparse.linalg as sslina
# import oct2py
import os
import subprocess
import zmq
import warnings
import timeit

import port
import plotfuns
import MOR

# oc = oct2py.Oct2Py()
# pRead=lambda path:ut.pRead(oc, path)
pRead=lambda path:ut.petscRead(path)

import os
os.environ["PATH"] = os.environ["PATH"] + ";D:\\Ordnerordner\\Software\\pythonEnvironments\\python3_10\\lib\\site-packages\\kaleido\\executable\\"


path=dict()
path['felis']     = os.path.abspath('').split('\\FELIS')[0]+'\\FELIS\\'
path['workDir'] = os.path.abspath('').split('\\interactive_mor_felis')[0]+'\\interactive_mor_felis\\'
path['plots']   = path['workDir']+'_Documentation\\_new_images\\'
path['mats']    = path['workDir']+'mats\\'
path['ports']   = path['mats']+'ports\\'
path['sols']    = path['mats']+'sols\\'
path['sysmats']    = path['mats']+'systemMats\\'


# modelName='accelerator_cavity'
nMax =50
f_data= {'fmin' : 8e9,
         'fmax' : 28e9,
         'nmor' : 1000,
         'ntest': 100,
         }
accuracy=1e-6

nPorts=2
nModes={'TB':0,
        'TS':0,
        'TE':10,
        'TM':10,}

cond=5.8e5

# launch_felis =True
init_felis   =True
recreate_mats=True
recreate_test=True

###############################################################################
###create constant matrices
# if launch_felis:
#     subprocess.call("H:\\FELIS_junction\\"+"felis "+modelName, shell=False)

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



timeStart=timeit.default_timer()
pod=MOR.Pod_adaptive(fAxis,ports,Mats,factors,RHS,JSrc,fIndsTest,sols_test,f_data['nmor'])
pod.set_residual_indices(np.linspace(0,np.shape(CC)[0]-1,int(np.shape(CC)[0]/1)).astype(int))
for iBasis in range(nMax):
    ###############################################################################
    #add new solutions:
    try:
        fnew=pod.select_new_freq()
    except Exception:
        warnings.warn('No more frequencies to select')
        break

    socket.send_string("solve: train_%d  %f" %(iBasis,fnew))
    message = socket.recv()
    print(message)
    newSol = pRead(path['sols'] +'train_'+str(iBasis))


    ###############################################################################
    #update
    # pod.update_Classic(newSol)
    pod.update_nested(newSol)
    # pod.print_time()
    ###############################################################################
    #post processing
    resTot,errTot,res_ROM,err_R_F=pod.get_conv()
    Znew=np.array(pod.get_Z()).astype('complex')

    res_Plot.append(resTot)
    err_Plot.append(errTot)
    res_curves.append(res_ROM)

    print('nBasis=%d remaining relative residual: %f' %(iBasis+1,resTot/np.max(res_Plot)))

    resDotsX.append(fAxis[solInds])
    resDotsY.append(res_ROM[solInds])
    Z.append(Znew)
    nBasiss.append(iBasis+1)

    #convergence check
    if resTot/np.max(res_Plot)<accuracy:
        break
timeMor=timeit.default_timer()-timeStart
print('MOR took %f seconds' %timeMor)


ZRef=np.zeros(len(fAxisTest)).astype('complex')
for i in range(len(fAxisTest)):
    ZRef[i]=MOR.impedance(sols_test[:,i], JSrc[:,fIndsTest[i]])


# fig=plotfuns.initPlot(title='err,res',logX=True,logY=True,xName='f',yName='val')
# plotfuns.plotLine (fig, fAxisOn, err_R_F,lineArgs={'name':'err','color':0})
# plotfuns.plotLine (fig, fAxisOn, res_ROM,lineArgs={'name':'res','color':1})
# plotfuns.showPlot(fig)
# raise Exception('stop')
plotconfig={'legendShow':True}

fig=plotfuns.initPlot(title='res convergence',logX=False,logY=True,xName='Number of Basis Functions',yName='')
plotfuns.plotLine (fig, nBasiss, res_Plot,lineArgs={'name':'res','color':0})
plotfuns.plotLine (fig, nBasiss, err_Plot,lineArgs={'name':'err','color':1})
plotfuns.showPlot(fig,show=True)
# plotfuns.exportPlot(fig, 'CubeWire_conv1', 'half', path=path['plots'],opts=plotconfig|{'yTick':3,'yRange':ut.listlog([1e-6,1e7]),'yFormat': '~e'})#,'xRange':ut.listlog([1,55]),'tickvalsX':[1,2,5,10,20,50,100]

# plotconfig={'legendShow':True,'xSuffix':''}
fig=plotfuns.initPlot(title='abs',logX=False,logY=True,xName='f in Hz',yName='Z in Ohm')
plotfuns.plotLine (fig, fAxis, np.abs(Z[19])  ,lineArgs={'name':'MOR_20','color':1} )
plotfuns.plotLine (fig, fAxis, np.abs(Z[49])  ,lineArgs={'name':'MOR_50','color':0} )
plotfuns.plotLine (fig, fAxis, np.abs(Z[-1])  ,lineArgs={'name':'MOR_%d' %len(Z),'color':2} )
plotfuns.plotLine (fig, fAxisTest, np.abs(ZRef) ,lineArgs={'name':'FEL','color':3,'dash':'dash'} )
plotfuns.showPlot(fig,show=True)
# plotfuns.exportPlot(fig, 'CubeWire_imp1', 'half', path=path['plots'],opts=plotconfig|{ 'legendPos':'botRight', 'xTick': 0.5e9, 'yTick': 1,'yRange':ut.listlog([0.02,50])})


inds,times,names =pod.get_time_for_plot()
fig=plotfuns.initPlot(title='time per iteration',logX=False,logY=False,xName='iteration',yName='time in s')
plotfuns.plotLines (fig, inds, times,curveName=names)
plotfuns.showPlot(fig,show=True)
# plotfuns.exportPlot(fig, 'acc_cav_time_NPvsQR', 'half', path=path['plots'],opts=plotconfig|{ 'legendPos':'topLeft', 'xTick': 5, 'yTick': 0.5,'yRange':[-0,1.8]})

#, 'xTick': 0.2, 'yTick': 1}











