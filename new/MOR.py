# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:16:01 2022

@author: Fred
"""

import subprocess
import timeit

import joblib as jl
import numpy as np
import numpy.linalg as nlina
import scipy.linalg as slina
import zmq
from myLibs import utility as ut

import misc
import port


def getFelisConstants():
    eps  = 8.8541878e-12
    mu = 1.2566371e-06
    c0 = 1/np.sqrt(mu*eps)
    return (eps,mu,c0) #(eps,mu,c0)=MOR.getFelisConstants()

def writeFAxis(fileName,values):
    with open(fileName, 'w') as f:
        f.write(str(len(values))+'\n')
        for i in range(len(values)):
            f.write(str(values[i])+'\n')


def createMatrices(felis_todos,pRead,path,cond,fmin,fmax,nTrain,nTest,nPorts,nModes):

    #  create fAxis
    fAxis = np.linspace(fmin, fmax, nTrain)

    #  delete old files
    if felis_todos['train']:
        subprocess.call("del /q /s " + path['sols'] + "\\train*", shell=True)
        subprocess.call("del /q /s " + path['sols'] + "\\freqs.csv", shell=True)
    else:
        fExisting = np.sort(ut.csvRead(path['sols'] + 'freqs.csv',delimiter=',')[:,0])
        fAxis= np.sort(np.append(fAxis,fExisting))
        nans=0
        # diff=np.zeros(len(fAxis)-1)
        ref=fAxis[0]
        for i in range(len(fAxis)-1):
            # diff[i]=fAxis[i+1]-fAxis[i]
            if fAxis[i+1]-fAxis[i]<1e-10*ref:
                fAxis[i]=np.NAN
                nans+=1
        if nans>0:
            fAxis=np.sort(fAxis)[0:-nans]

    if not felis_todos['exci']:
        # check if fAxis changed and rhs has to be recalculated
        fAxisChanged=False
        try:
            frhsOld=ut.csvRead(path['rhs'] + 'fAxis.csv',delimiter=',')[1:,0]
            if not np.allclose(frhsOld,fAxis,rtol=1e-10,atol=1e100):
                fAxisChanged=True
        except:
            ...
        if fAxisChanged:
            raise Exception('fAxis changed, redo RHS')

    #  Create socket to talk to server
    context = zmq.Context()
    misc.timeprint("Connecting to Felis")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    misc.timeprint("Connected to Felis")

    if felis_todos['init']:
        misc.timeprint("Initializing Felis")
        socket.send_string("initialize")
        message = socket.recv()
        misc.timeprint(message)

    #recreate matrices
    if felis_todos['mats']:
        subprocess.call("del /q " + path['mats'] + "\\*", shell=True)
        subprocess.call("del /q /s " + path['ports'] + "\\*", shell=True)
        #  export system matrices
        socket.send_string("export_mats")
        message = socket.recv()
        misc.timeprint(message)

        if cond > 0:
            socket.send_string("export_SIBc: %f" %fmin)
            message = socket.recv()
            misc.timeprint(message)

    if felis_todos['exci']:
        subprocess.call("del /q " + path['rhs'] + "\\*", shell=True)
        writeFAxis(path['rhs'] + 'fAxis.csv', fAxis)
        # socket.send_string("export_RHSs: %f, %f, %d" % (fmin, fmax, nTrain))
        socket.send_string("export_RHSs")
        message = socket.recv()
        misc.timeprint(message)

    ###############################################################################
    ###import/create constant matrices
    CC = pRead(path['mats'] + 'CC')
    MC = pRead(path['mats'] + 'MC')
    ME = pRead(path['mats'] + 'ME')
    if cond > 0:
        Sibc = pRead(path['mats'] + 'Sibc')
    else:
        Sibc = ME * 0

    ports = []
    for i in range(nPorts):
        ports.append(port.Port(path['ports'] + str(i) + '\\', pRead, nModes,fAxis))
        misc.timeprint('port read modes')
        ports[i].readModes()
        misc.timeprint('port read maps')
        ports[i].readMaps()
        misc.timeprint('port compute factors')
        ports[i].computeFactors()
        # misc.timeprint('port read EInc')
        # ports[i].readEInc()



    misc.timeprint('read RHS')
    RHS = pRead(path['rhs'] + 'RHSs').tocsc()  #Assumes that RHS is stored as a dense matrix or a 'full' sparse matrix
    misc.timeprint('read Js')
    JSrc = pRead(path['rhs'] + 'Js').tocsc() #Assumes that JSrc is stored as a dense matrix or a 'full' sparse matrix

    fAxisTest,fIndsTest = ut.closest_values(fAxis, np.linspace(fmin,fmax,nTest),returnInd=True)  # select test frequencies from fAxis
    # create test data

    if felis_todos['test']:
        subprocess.call("del /q /s " + path['sols'] + "\\test*", shell=True)

    runtimeTest=0
    for i in range(nTest):
        if felis_todos['test']:
            timeStart = timeit.default_timer()
            socket.send_string("solve: test_%d %f" % (i, fAxisTest[i]))
            message = socket.recv()
            runtimeTest+=timeit.default_timer() - timeStart
            misc.timeprint(message)
        if i == 0:
            sols_test = pRead(path['sols'] + 'test_0')
        else:
            sols_test = np.append(sols_test, pRead(path['sols'] + 'test_%d' % i), axis=1)
    misc.timeprint('runtime Felis for test data: %f' %runtimeTest)
    return (CC,ME,MC,Sibc,ports,RHS,JSrc,fAxis,fAxisTest,fIndsTest,sols_test,socket)



def impedance(u,j):
    return -np.dot(u,j.conj())


def nested_QR(U,R,a):
    if U==[]:
        U,R=slina.qr(a,mode='economic')
    else:
        U,R=slina.qr_insert(U,R,a,np.shape(U)[1],which='col')
    return (U,R)

class Pod_adaptive:
    def __init__(self,fAxis,ports,Mats,factors,RHS,JSrc,fIndsTest,SolsTest,nMax):
        self.fAxis=fAxis
        self.fIndsEval=list(range(len(fAxis)))
        self.ports=ports
        self.Mats=Mats
        self.factors=factors
        self.RHS=RHS
        self.fIndsTest=fIndsTest
        self.SolsTest=SolsTest
        self.JSrc=JSrc
        self.sols=[]
        self.U=[]
        self.R=[]
        self.res_ROM=np.zeros(len(fAxis))
        self.err_R_F=np.zeros(len(fIndsTest))
        self.Z=np.zeros(len(fAxis)).astype('complex')
        self.nMOR=len(fAxis)
        self.solInds=[]
        self.parallel=False
        self.times={
            'mat_assemble' :np.zeros((nMax,len(fAxis))),
            'port_assemble':np.zeros((nMax,len(fAxis))),
            'solve_LGS'    :np.zeros((nMax,len(fAxis))),
            'solve_proj'   :np.zeros((nMax,len(fAxis))),
            'err'          :np.zeros((nMax,len(fAxis))),
            'res_port'     :np.zeros((nMax,len(fAxis))),
            'res_mats'     :np.zeros((nMax,len(fAxis))),
        }
        self.resTMP=np.zeros(np.shape(Mats[0])[0]).astype('complex')
        self.uROM=None


    def update(self,newSol):
        if self.sols==[]:
            self.sols=newSol
            self.uROM=newSol[:,0]*0
        else:
            self.sols=np.append(self.sols, newSol, axis=1)
        self.nBasis=np.shape(self.sols)[1]

        self.U,self.R=nested_QR(self.U,self.R,newSol)

        Uh=self.U.conj().T

        self.Ushort=self.U[self.inds_u_proj,:]

        MatsR = []
        for i in range(len(self.Mats)):
            MatsR.append(Uh @ self.Mats[i] @ self.U)

        for i in range(len(self.ports)):
            self.ports[i].setU(self.U)
            self.ports[i].create_reduced_modeMats()

        n_jobs = 1
        if n_jobs== 1:
            for fInd in range(len(self.fAxis)):
                self.MOR_loop(MatsR, Uh, fInd)
        else:
            raise Exception('parallel currently disabled')
            jl.Parallel(n_jobs=n_jobs, prefer="threads")(jl.delayed(self.MOR_loop)(MatsR, Uh, fInd) for fInd in range(len(self.fAxis)))
        self.resTot = nlina.norm(self.res_ROM)


    def MOR_loop(self, MatsR, Uh, fInd):
        if not fInd in self.fIndsEval:
            return
        f = self.fAxis[fInd]

        start = timeit.default_timer()
        for i in range(len(self.ports)):
            if i == 0:
                portMat = self.ports[i].getReducedPortMat(fInd)
            else:
                portMat += self.ports[i].getReducedPortMat(fInd)
        self.add_time('port_assemble', start,fInd)
        # end port matrix creation

        start = timeit.default_timer()
        for i in range(len(self.Mats)):
            if i == 0:
                AROM = MatsR[i] * self.factors[i](f)
            else:
                AROM += MatsR[i] * self.factors[i](f)
        AROM += portMat
        self.add_time('mat_assemble', start,fInd)


        # start=timeit.default_timer()
        # self.add_time('projectR', start)

        start = timeit.default_timer()
        rhs_red = Uh @ self.getRHS(fInd)
        AQ, AR = slina.qr(AROM)
        vMor = slina.solve_triangular(AR, AQ.conj().T @ rhs_red)
        self.add_time('solve_LGS', start,fInd)

        start = timeit.default_timer()
        self.uROM[self.inds_u_proj]=self.Ushort @ vMor
        # self.uROM[self.inds_u_proj] = (self.U[self.inds_u_proj,:] @ vMor)  # [:,0]
        # self.uROM = (self.U @ vMor)  # [:,0]
        self.add_time('solve_proj', start,fInd)
        if fInd in self.fIndsTest:
            i = np.where(self.fIndsTest == fInd)[0][0]
            self.err_R_F[i] = nlina.norm(self.uROM - self.SolsTest[:, i])

        self.residual_estimation(fInd)

        # postprocess solution: impedance+sParams
        self.Z[fInd] = impedance(self.uROM, self.getJSrc(fInd))
        # ZFM[fInd]=impedance(SolsOn[:,fInd],JSrcOn[:,fInd])

    def sParam(self,u,f):
        for i in range(len(self.ports)):
            S=self.ports[i].getSParams(u,f)


    def set_residual_indices(self,residual_indices):
        self.residual_indices=residual_indices
        self.Mats_res=[]
        for i in range(len(self.Mats)):
            self.Mats_res.append(self.Mats[i][residual_indices,:])
        tempMat=np.sum(self.Mats_res)
        self.inds_u_res=np.unique(tempMat.indices)
        for i in range(len(self.Mats)):
            self.Mats_res[i]=self.Mats_res[i][:,self.inds_u_res]

    def set_projection_indices(self):
        """
        Set the indices of the projection matrix that are used to project the reduced basis onto the solution. These
        indices constist of the residual indices and the indices of the beam.
        Returns
        -------

        """
        inds_u_res=self.inds_u_res
        inds_beam=self.RHS[:,0].indices
        inds_port=self.ports[0].get3Dinds()
        for i in range(1,len(self.ports)):
            inds_port=np.append(inds_port,self.ports[i].get3Dinds())
        self.inds_u_proj=np.sort(np.unique(np.concatenate((inds_u_res,inds_beam,inds_port))))

    def residual_estimation(self, fInd):
        f = self.fAxis[fInd]
        start = timeit.default_timer()
        uROM_short = self.uROM[self.inds_u_res]
        for i in range(len(self.Mats)):
            if i == 0:
                RHSrom = self.Mats_res[i] @ (self.factors[i](f) * uROM_short)
            else:
                RHSrom += self.Mats_res[i] @ (self.factors[i](f) * uROM_short)
        self.add_time('res_mats', start,fInd)

        start = timeit.default_timer()
        for i in range(len(self.ports)):
            self.resTMP[self.ports[i].get3Dinds()]=self.ports[i].multiplyVecPortMatSparse(self.uROM,fInd)
            RHSrom += self.resTMP[self.residual_indices]
            self.resTMP[self.ports[i].get3Dinds()] *=0
        self.add_time('res_port', start,fInd)

        self.res_ROM[fInd] = nlina.norm(self.getRHS(fInd)[self.residual_indices] - RHSrom)/np.sqrt(len(self.residual_indices))

    def add_time(self, timeName, start,fInd):
        self.times[timeName][self.nBasis-1,fInd]+=(timeit.default_timer() - start)

    def get_time_for_plot(self):
        times = []
        names = []
        for i in range(len(self.times)):
            times.append([])
            names.append(list(self.times.keys())[i])
            times[-1]=np.sum(self.times[names[i]],axis=1)
        inds = np.cumsum(np.ones(np.shape(times)[1]))
        return inds,times,names



    def fAxisGreedy(self,density):
        self.fIndsEval=np.round(np.linspace(0,len(self.fAxis)-1,int(len(self.fAxis)*density))).astype(int)

    def fAxisGreedy_treeRefine(self, refine_ind):
        ind_fIndsEval = np.where(self.fIndsEval == refine_ind)[0][0]
        if ind_fIndsEval+1 <= len(self.fIndsEval)-1:
            if self.fIndsEval[ind_fIndsEval+1]-self.fIndsEval[ind_fIndsEval]>1:
                indNew= np.round((self.fIndsEval[ind_fIndsEval]+self.fIndsEval[ind_fIndsEval+1])/2).astype(int)
                self.fIndsEval=np.insert(self.fIndsEval,ind_fIndsEval+1,indNew)
            else:
                misc.timeprint('no refinement possible for higher frequency')
        if ind_fIndsEval-1 >= 0:
            if self.fIndsEval[ind_fIndsEval]-self.fIndsEval[ind_fIndsEval-1]>1:
                indNew= np.round((self.fIndsEval[ind_fIndsEval]+self.fIndsEval[ind_fIndsEval-1])/2).astype(int)
                self.fIndsEval=np.insert(self.fIndsEval,ind_fIndsEval,indNew)
            else:
                misc.timeprint('no refinement possible for lower frequency')

    def select_new_freq_greedy(self):
        if len(self.solInds) == 0:
            a=int(len(self.fIndsEval) / 2)
            newSolInd=self.fIndsEval[a]
            self.solInds.append(newSolInd)
        else:
            facts= np.array([0 if i in self.solInds else 1 for i in range(len(self.fAxis))])
            if np.sum(facts)==0:
                raise Exception('no more frequencies available')
            res = (self.res_ROM*facts)[self.fIndsEval]
            newSolInd= self.fIndsEval[np.argmax(res)]
            self.solInds.append(newSolInd)
            self.fAxisGreedy_treeRefine(newSolInd)
        return self.fAxis[newSolInd]

    def select_new_freq(self):
        if len(self.solInds) == 0:
            self.solInds.append((len(self.fAxis) / 2).astype(int))
        else:
            facts = np.array([0 if i in self.solInds else 1 for i in range(len(self.fAxis))])
            if np.sum(facts) == 0:
                raise Exception('no more frequencies available')
            self.solInds.append(np.argmax(self.res_ROM * facts))
        return self.fAxis[self.solInds[-1]]

    def register_new_freq(self, freq):
        self.solInds.append(np.where(np.abs((self.fAxis - freq))<(freq/1e10))[0][0])
        self.fIndsEval=np.sort(np.append(self.fIndsEval,self.solInds[-1]))
        self.fAxisGreedy_treeRefine(self.solInds[-1])
    
    def all_freqs(self):
        self.fIndsEval=np.array(range(len(self.fAxis)))


    def get_conv(self):
        resTot=np.linalg.norm(self.res_ROM[self.fIndsEval])
        errTot=np.linalg.norm(self.err_R_F)
        return resTot,errTot,self.res_ROM[self.fIndsEval],self.err_R_F

    def get_Z(self):
        return self.fAxis[self.fIndsEval],self.Z[self.fIndsEval]

    def getRHS(self,fInd):
        return self.RHS[:,fInd].toarray()[:,0]

    def getJSrc(self,fInd):
        return self.JSrc[:,fInd].toarray()[:,0]
    
    def get_fAxis(self):
        return self.fAxis






    
    
    
    
    
    