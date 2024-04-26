


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
import configs

import h5py

# oc = oct2py.Oct2Py()
# pRead=lambda path:ut.pRead(oc, path)
pRead=lambda path:ut.petscRead(path+'.pmat')



import os
os.environ["PATH"] = os.environ["PATH"] + ";D:\\Ordnerordner\\Software\\pythonEnvironments\\python3_10\\lib\\site-packages\\kaleido\\executable\\"


path=dict()
path['felis_projects']     = os.path.abspath('').split('\\FELIS_Projects')[0]+'\\FELIS_Projects\\'
path['workDir'] = os.path.abspath('')+'\\'
path['felis_bin']= path['felis_projects']+"FELIS_DDM_IMP\\"
path['mats']    = path['felis_bin']+'mats\\'
# path['mats']    = path['felis_bin']+'interactive_mor_felis\\mats\\'
path['ports']   = path['mats']+'ports\\'
path['sols']    = path['mats']+'sols\\'
path['rhs']    = path['mats']+'rhs\\'
path['sysmats']    = path['mats']+'systemMats\\'
path['plots']   = path['workDir']+'_new_images\\'

#square cavity
configread=configs.roundLossyWG_config

##undulator
# configread=configs.sh12_3b_config


[nMax,NChecks,n_MC,frac_Greedy,accuracy,f_data,symmetry,nPorts,exportZ,nModes,orderHex,orderTet,runId,felis_todos,dbName,saveMemory,plotConv,plotTime,plotZls,plotZts]=configread()
# launch_felis =True
###############################################################################
###create constant matrices
# if launch_felis:
#     subprocess.call("H:\\FELIS_junction\\"+"felis "+modelName, shell=False)

time_start_program=timeit.default_timer()
(facs,Mats,ports,RHS,beams,fAxis,fAxisTest,fIndsTest,sols_test,Z_felis,socket)=\
    MOR.createMatrices(felis_todos,pRead,path,f_data['fmin'],f_data['fmax'],f_data['nmor'],f_data['ntest'],nPorts,nModes)
f_data['nmor']=len(fAxis)



###############################################################################
###MOR Solving

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
Zs=[]
fAxes=[]

time_setup=timeit.default_timer()-time_start_program
timeStart=timeit.default_timer()
pod=MOR.Pod_adaptive(fAxis,ports,facs,Mats,RHS,beams,symmetry,fIndsTest,sols_test,nMax)
sol=pod.directSolveAtTest(fIndsTest[0])
err=np.linalg.norm(sols_test[:,0]-sol)
tmp=np.array([sol, sols_test[:,0]])
impedance=MOR.impedance(sol,beams.getJ(0)[:,fIndsTest[0]].toarray(),1)
impedanceRef=MOR.impedance(sols_test[:,0],beams.getJ(0)[:,fIndsTest[0]].toarray(),1)

w = lambda freq: 2 * np.pi * freq
fac = lambda f: -(w(f) / MOR.getFelisConstants()[2]) ** 2
print('errSol'+str(err))
print('computed impedance='+str(abs(impedance))+' reference impedance='+str(abs(impedanceRef)))
print('norm CC:' + str(sslina.norm(Mats[0],'fro')))
print('norm ME:' + str(sslina.norm(Mats[1],'fro')))
print('factor ME'+str(fac(fAxis[fIndsTest[0]])))
print('2norm rhs: '+str(nlina.norm(RHS[:,fIndsTest[0]].toarray())))





raise Exception('Program end')

if saveMemory:
    pod.deactivate_timing()
pod.set_residual_indices(np.linspace(0,np.shape(Mats[0])[0]-1, n_MC).astype(int) ,int(np.shape(Mats[0])[0]/n_MC))
pod.set_projection_indices()
pod.fAxisGreedy(1/frac_Greedy)
for iBasis in range(nMax):
    ###############################################################################

    #add new solutions:
    if os.path.isfile(path['sols'] +'train_'+str(iBasis)+'.pmat'):
        timeFem=0
        fnew=misc.csv_readLine(path['sols'] +'freqs',iBasis)[0]
        pod.register_new_freq(fnew)
    else:
        start = timeit.default_timer()
        fnew=pod.select_new_freq_greedy()
        socket.send_string("solve: train_%d  %f" %(iBasis,fnew))
        message = socket.recv()
        misc.timeprint(message)
        timeFem=timeit.default_timer() - start
        misc.csv_writeLine(path['sols'] +'freqs',fnew,iBasis)
    newSol = pRead(path['sols'] +'train_'+str(iBasis))



    if final:
        pod.all_freqs()
    ###############################################################################
    #update
    # pod.update_Classic(newSol)
    startTotal = timeit.default_timer()
    pod.update(newSol)
    pod.add_time('FEM', timeit.default_timer()-timeFem, 0)
    pod.add_time('MOR', startTotal,0)

    # pod.print_time()
    ###############################################################################
    #post processing
    resTot,errTot,res_ROM,err_R_F=pod.get_conv()

    fAxis_new,Zsnew=pod.get_Zl()


    res_Plot.append(resTot)
    err_Plot.append(errTot)

    misc.timeprint('nBasis=%d remaining relative residual: %f, err: %f' %(iBasis+1,resTot,errTot)) #/np.max(res_Plot)

    if not saveMemory:
        res_curves.append(res_ROM)
        resDotsX.append(fAxis[solInds])
        resDotsY.append(res_ROM[solInds])
        Zs.append(Zsnew)
        fAxes.append(fAxis_new)
    else:
        Zs=[Zsnew]
        fAxes=[fAxis_new]

    nBasiss.append(iBasis+1)

    #convergence check
    if final: break

    if errTot<accuracy:
        nChecks-=1
        print('Accuracy reached, nChecks reduced to '+str(nChecks))
    else:
        nChecks=NChecks

    if nChecks==0 or iBasis==nMax-2:
        final=True
        misc.timeprint('last iteration')

Zts=[]
# np.zeros(len(fAxis_new)).astype('complex')
for iBeam in range(beams.getN()):
    if iBeam==beams.getDrivingBeam():
        continue
    Ztnew={'pos':beams.getPosition(iBeam)}
    Ztnew['vals']=pod.get_Zt(iBeam)
    Ztnew['index'] = iBeam
    Zts.append(Ztnew)


timeMor=timeit.default_timer()-timeStart
print('MOR took %f seconds' %timeMor)

Zrefs=[]
for iBeam in range(len(Zsnew)):
    ZRef=np.zeros(len(fAxisTest)).astype('complex')
    for j in range(len(fAxisTest)):
        Znew=MOR.impedance(sols_test[:,j], beams.getJ(iBeam)[:,fIndsTest[j]].toarray()[:,0],symmetry)
        ZRef[j]=Znew
    Zrefs.append(ZRef)
misc.timeprint('Solver end')

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
plotfuns.showPlot(fig,show=plotConv)
# plotfuns.exportPlot(fig, 'CubeWire_conv1', 'half', path=path['plots'],opts=plotconfig|{'yTick':3,'yRange':ut.listlog([1e-6,1e7]),'yFormat': '~e'})#,'xRange':ut.listlog([1,55]),'tickvalsX':[1,2,5,10,20,50,100]


for plotZl in plotZls:
    if plotZl=='a':
        absReIm=lambda x: np.abs(x)
        logY=True
        title='abs'
    elif plotZl=='r':
        absReIm=lambda x: np.real(x)
        logY=False
        title='re'
    elif plotZl=='i':
        absReIm=lambda x: np.imag(x)
        logY=False
        title='im'
    else:
        continue
    plotconfig={'legendShow':True,'xSuffix':''}
    fig=plotfuns.initPlot(title=title,logX=False,logY=logY,xName='f in Hz',yName='Zl in Ohm')
    for iBeam in range(beams.getN()):

        plotfuns.plotLine (fig, fAxes[-1], absReIm(Zs[-1][iBeam])   ,lineArgs={'name':'Beam %d, It. %d' %(iBeam,len(Zs)),'color':iBeam} )
        plotfuns.plotCloud (fig, fAxisTest, absReIm(Zrefs[iBeam])   ,cloudArgs={'name':'REF %d'%(iBeam),'color':plotfuns.colorList(colIndex=iBeam),'size':15} )
        plotfuns.plotCloud (fig, fAxisTest, absReIm(Z_felis[iBeam,:]),cloudArgs={'name':'FEL %d'%(iBeam),'color':plotfuns.colorList(colIndex=iBeam),'size':5} )
    plotfuns.showPlot(fig,show=True)
    # plotfuns.exportPlot(fig, 'SH12_imp_Ord2_sibc_1', 'full', path=path['plots'],opts=plotconfig|{ 'legendPos':'topRight', 'xTick': 0.1e9, 'yTick': 1,'yRange':[0.2,2000]})


for plotZt in plotZts:
    if plotZt=='a':
        absReIm=lambda x: np.abs(x)
        logY=True
        title='abs'
    elif plotZt=='r':
        absReIm=lambda x: np.real(x)
        logY=False
        title='re'
    elif plotZt=='i':
        absReIm=lambda x: np.imag(x)
        logY=False
        title='im'
    else:
        continue
    plotconfig={'legendShow':True,'xSuffix':''}
    fig=plotfuns.initPlot(title=title,logX=False,logY=logY,xName='f in Hz',yName='Zt in Ohm/m')
    for iBeam in range(beams.getN()-1):
        plotfuns.plotLine (fig, fAxes[-1], absReIm(Zts[iBeam]['vals'])   ,lineArgs={'name':'Beam %d, It. %d' %(Zts[iBeam]['index'],len(Zs)),'color':iBeam} )
    plotfuns.showPlot(fig,show=True)
    # plotfuns.exportPlot(fig, 'SH12_imp_Ord2_sibc_1', 'full', path=path['plots'],opts=plotconfig|{ 'legendPos':'topRight', 'xTick': 0.1e9, 'yTick': 1,'yRange':[0.2,2000]})



if not saveMemory:
    inds,times,names =pod.get_time_for_plot()
    tD=dict(zip(names,times))
    try:
        firstInd=np.where(tD['mat_assemble']==0)[0][0]
        times=np.array(times)[:,:firstInd]
        inds=inds[:firstInd]
    except Exception:
        ...
    namesPlt=['Total','FEM','MOR_total','MOR_solve','MOR_res.','Misc.']
    timesPlt=[tD['sum'],
              tD['FEM'],
              -1,
              tD['mat_assemble']+tD['port_assemble']+tD['preparations1']+tD['preparations2']+tD['preparations3']+tD['solve_LGS_QR']+tD['project_RHS'],
              tD['solve_proj']+tD['res_port']+tD['res_mats']+tD['res_norm'],
              tD['err']+tD['Z']+tD['misc']
            ]
    timesPlt[2]=timesPlt[3]+timesPlt[4]
    fig=plotfuns.initPlot(title='time per iteration',logX=False,logY=False,xName='iteration',yName='time in s')
    plotfuns.plotLines (fig, inds, timesPlt,curveName=namesPlt)
    plotfuns.showPlot(fig,show=plotTime)
    # # plotfuns.exportPlot(fig, 'acc_cav_time_NPvsQR', 'half', path=path['plots'],opts=plotconfig|{ 'legendPos':'topLeft', 'xTick': 5, 'yTick': 0.5,'yRange':[-0,1.8]})

# #, 'xTick': 0.2, 'yTick': 1}

for iBeam in range(beams.getN()):
    print("Beam %d, " %iBeam, end='')
    print("pos (x,y)= "+str(beams.getPosition(iBeam)), end='')
    if iBeam==beams.getDrivingBeam():
        print(" is driving beam", end='')
    print('')



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
            grp.create_dataset('test/fAxis', data=fAxisTest)
            for iBeam in range(beams.getN()):
                grpBeam=grp.create_group(str(iBeam))
                grpBeam.create_dataset('ZRe'  , data=np.real(Zs[-1][iBeam]))
                grpBeam.create_dataset('ZIm'  , data=np.imag(Zs[-1][iBeam]))
                grpBeam.create_dataset('ZAbs' , data=np.abs (Zs[-1][iBeam]))
                grpBeam.create_dataset('pos' , data=beams.getPosition(iBeam))
                grpBeam.create_dataset('test/ZRe'  , data=np.real(Zrefs[iBeam]))
                grpBeam.create_dataset('test/ZIm'  , data=np.imag(Zrefs[iBeam]))
                grpBeam.create_dataset('test/ZAbs' , data=np.abs (Zrefs[iBeam]))
            grp.create_dataset('drivingBeam' , data=beams.getDrivingBeam())
            grp.create_dataset('NBeams' , data=beams.getN())
            grp.create_dataset('acc'  , data=accuracy)
            grp.create_dataset('n_mc'  , data=n_MC)
            grp.create_dataset('DoFs'  , data=np.shape(CC)[0])
            grp.create_dataset('frac_Sparse'  , data=n_MC/np.shape(CC)[0])
            grp.create_dataset('frac_Greedy'  , data=frac_Greedy)
            grp.create_dataset('time_times' , data=timesPlt)
            grp.create_dataset('time_names' , data=namesPlt)
            grp.create_dataset('time_inds' , data=inds)
            grp.create_dataset('time_setup' , data=time_setup)
            grp.create_dataset('time_total' , data=time_setup+timeMor)
            grp.create_dataset('err' , data=err_Plot)
            grp.create_dataset('res' , data=res_Plot)

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


















