


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
from myLibs import utility as ut
from myLibs import plotfuns
import MOR
import misc

import h5py

# oc = oct2py.Oct2Py()
# pRead=lambda path:ut.pRead(oc, path)
pRead=lambda path:ut.petscRead(path+'.pmat')
def filterZ(Z,doFiltering):
    if not doFiltering:
        return Z
    return Z-np.real(Z[0])


import os
os.environ["PATH"] = os.environ["PATH"] + ";D:\\Ordnerordner\\Software\\pythonEnvironments\\python3_10\\lib\\site-packages\\kaleido\\executable\\"


path=dict()
path['felis_projects']     = os.path.abspath('').split('\\FELIS_Projects')[0]+'\\FELIS_Projects\\'
path['workDir'] = os.path.abspath('')+'\\'
path['felis_bin']= path['felis_projects']+"FELIS_Binary\\"
path['mats']    = path['felis_bin']+'mats\\'
# path['mats']    = path['felis_bin']+'interactive_mor_felis\\mats\\'
path['ports']   = path['mats']+'ports\\'
path['sols']    = path['mats']+'sols\\'
path['rhs']    = path['mats']+'rhs\\'
path['sysmats']    = path['mats']+'systemMats\\'
path['plots']   = path['workDir']+'_new_images\\'


#######MOR CONFIG
nMax = 30
f_data= {'fmin' : 0.08e9,
         'fmax' : 0.2e9,
         'nmor' : 100,
         'ntest': 10,          # 1577.186820
         }

frac_Greedy=8
frac_Sparse=16
accuracy=1e-6
NChecks=3
doFiltering=False

plotZs=['a'] #a,r,i

exportZ=False
dbName='SH12sibc_Mor_1'
orderHex=0
orderTet=0
runId=0


felis_todos=dict()
felis_todos['init'] =False
felis_todos['mats'] =False
felis_todos['exci'] =False
felis_todos['test'] =False
felis_todos['train']=False

#########MODEL CONFIG
symmetry=2

nPorts=2
nModes={'TB':0,
        'TS':0,
        'TE':0,
        'TM':1,}

cond=1.4e6 #sh12
# cond=5.8e+3 #cubewire
cond=0 #5.8e5

# launch_felis =True
###############################################################################
###create constant matrices
# if launch_felis:
#     subprocess.call("H:\\FELIS_junction\\"+"felis "+modelName, shell=False)

(CC,ME,MC,Sibc,ports,RHS,JSrc,fAxis,fAxisTest,fIndsTest,sols_test,Z_felis,socket)=\
    MOR.createMatrices(felis_todos,pRead,path,cond,f_data['fmin'],f_data['fmax'],f_data['nmor'],f_data['ntest'],nPorts,nModes)
f_data['nmor']=len(fAxis)


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
    lambda f:factorSibc(f)#/factorSibc(fAxis[0])
        ]




nChecks=NChecks
final=False
res_Plot=[]
err_Plot=[]
res_curves=[]
resDotsX=[]
resDotsY=[]
nBasiss=[]
res_ROM=0
solInds=[]
Z=[]
fAxes=[]

timeStart=timeit.default_timer()
pod=MOR.Pod_adaptive(fAxis,ports,Mats,factors,RHS,JSrc,symmetry,fIndsTest,sols_test,nMax)
if cond>0:
    pod.correct_sibc(3)
pod.set_residual_indices(np.linspace(0,np.shape(CC)[0]-1,int(np.shape(CC)[0]/frac_Sparse)).astype(int))
pod.set_projection_indices()
pod.fAxisGreedy(1/frac_Greedy)
for iBasis in range(nMax):
    ###############################################################################

    #add new solutions:
    if os.path.isfile(path['sols'] +'train_'+str(iBasis)+'.pmat'):
        fnew=misc.csv_readLine(path['sols'] +'freqs',iBasis)[0]
        pod.register_new_freq(fnew)
    else:
        fnew=pod.select_new_freq_greedy()
        socket.send_string("solve: train_%d  %f" %(iBasis,fnew))
        message = socket.recv()
        misc.timeprint(message)
        misc.csv_writeLine(path['sols'] +'freqs',fnew,iBasis)
    newSol = pRead(path['sols'] +'train_'+str(iBasis))



    if final:
        pod.all_freqs()
    ###############################################################################
    #update
    # pod.update_Classic(newSol)
    pod.update(newSol)
    # pod.print_time()
    ###############################################################################
    #post processing
    resTot,errTot,res_ROM,err_R_F=pod.get_conv()
    fAxis_new,Znew=pod.get_Z()

    res_Plot.append(resTot)
    err_Plot.append(errTot)
    res_curves.append(res_ROM)

    misc.timeprint('nBasis=%d remaining relative residual: %f, err: %f' %(iBasis+1,resTot,errTot)) #/np.max(res_Plot)

    resDotsX.append(fAxis[solInds])
    resDotsY.append(res_ROM[solInds])
    Znew=filterZ(Znew,doFiltering)
    Z.append(Znew)
    fAxes.append(fAxis_new)
    nBasiss.append(iBasis+1)

    #convergence check
    if final: break

    if resTot<accuracy:
        nChecks-=1
        print('Accuracy reached, nChecks reduced to '+str(nChecks))
    else:
        nChecks=NChecks

    if nChecks==0:
        final=True
        misc.timeprint('last iteration')
timeMor=timeit.default_timer()-timeStart
print('MOR took %f seconds' %timeMor)


ZRef=np.zeros(len(fAxisTest)).astype('complex')
for i in range(len(fAxisTest)):
    Znew=MOR.impedance(sols_test[:,i], JSrc[:,fIndsTest[i]].toarray()[:,0],symmetry)
    ZRef[i]=Znew
ZRef=filterZ(ZRef, doFiltering)
##############################################################################
# single Pod Plots
fig=plotfuns.initPlot(title='err,res',logX=True,logY=True,xName='f',yName='val')
plotfuns.plotLine (fig, fAxis, err_R_F,lineArgs={'name':'err','color':0})
plotfuns.plotLine (fig, fAxis, res_ROM,lineArgs={'name':'res','color':1})
plotfuns.showPlot(fig,show=False)
# raise Exception('stop')

plotconfig={'legendShow':True}

fig=plotfuns.initPlot(title='res convergence',logX=False,logY=True,xName='Number of Basis Functions',yName='')
plotfuns.plotLine (fig, nBasiss, res_Plot,lineArgs={'name':'res','color':0})
plotfuns.plotLine (fig, nBasiss, err_Plot,lineArgs={'name':'err','color':1})
plotfuns.showPlot(fig,show=True)
# plotfuns.exportPlot(fig, 'CubeWire_conv1', 'half', path=path['plots'],opts=plotconfig|{'yTick':3,'yRange':ut.listlog([1e-6,1e7]),'yFormat': '~e'})#,'xRange':ut.listlog([1,55]),'tickvalsX':[1,2,5,10,20,50,100]


for plotZ in plotZs:
    if plotZ=='a':
        absReIm=lambda x: np.abs(x)
        logY=True
        title='abs'
    elif plotZ=='r':
        absReIm=lambda x: np.real(x)
        logY=False
        title='re'
    elif plotZ=='i':
        absReIm=lambda x: np.imag(x)
        logY=False
        title='im'
    else:
        continue
    plotconfig={'legendShow':True,'xSuffix':''}
    plotInds=[]
    fig=plotfuns.initPlot(title=title,logX=False,logY=logY,xName='f in Hz',yName='Z in Ohm')
    for i in range(len(plotInds)):
        plotfuns.plotLine (fig, fAxes[plotInds[i]-1], absReIm(Z[plotInds[i]-1]),lineArgs={'name':'MOR_%d' %plotInds[i],'color':i+1} )
    plotfuns.plotLine (fig, fAxes[-1], absReIm(Z[-1])   ,lineArgs={'name':'MOR_%d' %len(Z),'color':1} )
    # plotfuns.plotLine (fig, fAxisTest, np.abs(ZRef)/symmetry**2 ,lineArgs={'name':'FEL','color':3,'dash':'dash'} )
    plotfuns.plotCloud (fig, fAxisTest, absReIm(ZRef)   ,cloudArgs={'name':'REF','color':'red','size':12} )
    plotfuns.plotCloud (fig, fAxisTest, absReIm(Z_felis),cloudArgs={'name':'FEL','color':'green','size':6} )
    # plotfuns.plotLine (fig, fAxisTest, ref,lineArgs={'name':'FEL','color':4,'dash':'dash'} )
    plotfuns.showPlot(fig,show=True)
    # plotfuns.exportPlot(fig, 'SH12_imp_Ord2_sibc_1', 'full', path=path['plots'],opts=plotconfig|{ 'legendPos':'topRight', 'xTick': 0.1e9, 'yTick': 1,'yRange':[0.2,2000]})


inds,times,names =pod.get_time_for_plot()
fig=plotfuns.initPlot(title='time per iteration',logX=False,logY=False,xName='iteration',yName='time in s')
plotfuns.plotLines (fig, inds, times,curveName=names)
plotfuns.showPlot(fig,show=True)
# # plotfuns.exportPlot(fig, 'acc_cav_time_NPvsQR', 'half', path=path['plots'],opts=plotconfig|{ 'legendPos':'topLeft', 'xTick': 5, 'yTick': 0.5,'yRange':[-0,1.8]})

# #, 'xTick': 0.2, 'yTick': 1}



# a=np.sum(np.abs(JSrc))

###############################################################################
##Data export

if  exportZ:
    meshId=0
    key = "%d/%d.%d/%d" % (meshId, orderHex,orderTet, runId)
    with h5py.File(dbName+'.hdf5', 'a') as f:
            if key in f.keys():
                x = input('group already exists, overwrite? (y/n)')
                if x == 'y':
                    del f[key]
                else:
                    raise Exception('group already exists')
            grp = f.create_group(key)
            grp.create_dataset('fAxis', data=fAxes[-1])
            grp.create_dataset('ZRe'  , data=np.real(Z[-1]))
            grp.create_dataset('ZIm'  , data=np.imag(Z[-1]))
            grp.create_dataset('ZAbs' , data=np.abs (Z[-1]))
            grp.create_dataset('acc'  , data=accuracy)
else:
    print('Z not exported!')




# def impedance(u,j,sym):
#     return -np.vdot(j,u)/sym
#     # return -np.dot(u,j.conj())/symmetry
#
# ZRef=np.zeros(len(fAxisTest)).astype('complex')
# for i in range(len(fAxisTest)):
#     Znew=impedance(sols_test[:,i], JSrc[:,fIndsTest[i]].toarray()[:,0],symmetry)
#     ZRef[i]=Znew
#
# # ZRef/Z_felis
# np.real(ZRef)/np.real(Z_felis)
# # np.imag(ZRef)/np.imag(Z_felis)





























