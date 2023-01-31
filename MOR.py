# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:16:01 2022

@author: Fred
"""

import subprocess
import timeit
import warnings

import numpy as np
import numpy.linalg as nlina
import scipy.linalg as slina
import zmq

import port
import utility as ut
import joblib as jl


def getFelisConstants():
    eps  = 8.8541878e-12
    mu = 1.2566371e-06
    c0 = 1/np.sqrt(mu*eps)
    return (eps,mu,c0) #(eps,mu,c0)=MOR.getFelisConstants()

def createMatrices(init_felis,recreate_mats,recreate_test,pRead,path,cond,fmin,fmax,nTrain,nTest,nPorts,nModes):

    #  create fAxis
    fAxis = np.linspace(fmin, fmax, nTrain)

    #  delete old files
    subprocess.call("del /q /s " + path['sols'] + "\\train*", shell=True)

    #  Create socket to talk to server
    context = zmq.Context()
    print("Connecting to Felis")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    if init_felis:
        socket.send_string("initialize")
        message = socket.recv()
        print(message)

    #recreate matrices
    if recreate_mats:
        subprocess.call("del /q /s " + path['mats'] + "\\*", shell=True)
        #  export system matrices
        socket.send_string("export_mats")
        message = socket.recv()
        print(message)

        if cond > 0:
            socket.send_string("export_SIBc: %f" %fmin)
            message = socket.recv()
            print(message)

        socket.send_string("export_RHSs: %f, %f, %d" % (fmin, fmax, nTrain))
        message = socket.recv()
        print(message)

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
        print('port read modes')
        ports[i].readModes()
        print('port read maps')
        ports[i].readMaps()
        print('port compute factors')
        ports[i].computeFactors()


    print('read RHS')
    RHS = pRead(path['mats'] + 'RHSs')  #Assumes that RHS is stored as a dense matrix or a 'full' sparse matrix
    print('read Js')
    JSrc = pRead(path['mats'] + 'Js')  #Assumes that JSrc is stored as a dense matrix or a 'full' sparse matrix

    fAxisTest,fIndsTest = ut.closest_values(fAxis, np.linspace(fmin,fmax,nTest),returnInd=True)  # select test frequencies from fAxis
    # create test data

    if recreate_mats and not recreate_test:
        recreate_test=True
        warnings.warn('if recreate_mats is True, recreate_test also has to be True')

    if recreate_test:
        subprocess.call("del /q /s " + path['sols'] + "\\test*", shell=True)

    runtimeTest=0
    for i in range(nTest):
        if recreate_test:
            timeStart = timeit.default_timer()
            socket.send_string("solve: test_%d %f" % (i, fAxisTest[i]))
            message = socket.recv()
            runtimeTest+=timeit.default_timer() - timeStart
            print(message)
        if i == 0:
            sols_test = pRead(path['sols'] + 'test_0')
        else:
            sols_test = np.append(sols_test, pRead(path['sols'] + 'test_%d' % i), axis=1)
    print('runtime Felis for test data: %f' %runtimeTest)
    return (CC,ME,MC,Sibc,ports,RHS,JSrc,fAxis,fAxisTest,fIndsTest,sols_test,socket)

def selectIndAdaptive(nFreqs,res,solInds):
    if len(solInds) == 0:
        solInds.append(np.round(nFreqs/2).astype(int))
    else:
        facts=np.array([0 if i in solInds else 1 for i in range(nFreqs)])
        if np.sum(facts)==0:
            raise Exception('no more frequencies available')
        solInds.append(np.argmax(res*facts))

def impedance(u,j):
    return -np.dot(u,j.conj())

def nested_QR(U,R,a):
    if U==[]:
        U,R=slina.qr(a,mode='economic')
    else:
        U,R=slina.qr_insert(U,R,a,np.shape(U)[1],which='col')
    return (U,R)

class Pod_adaptive:
    def __init__(self,fAxis,ports,Mats,factors,RHS,JSrc,fIndsTest,SolsTest,nMOR,nMax):
        self.fAxis=fAxis
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
        self.nMOR=nMOR
        self.solInds=[]

        self.times={
            'mat_assemble' :np.zeros((nMax,len(fAxis))),
            'port_assemble':np.zeros((nMax,len(fAxis))),
            'solve_LGS'    :np.zeros((nMax,len(fAxis))),
            'solve_proj'   :np.zeros((nMax,len(fAxis))),
            'err'          :np.zeros((nMax,len(fAxis))),
            'res_port'     :np.zeros((nMax,len(fAxis))),
            'res_mats'     :np.zeros((nMax,len(fAxis))),
            'time_step'    :np.zeros(nMax),
        }


    def update(self,newSol):
        if self.sols==[]:
            self.sols=newSol
        else:
            self.sols=np.append(self.sols, newSol, axis=1)
        nBasis=len(self.sols)

        self.U,self.R=nested_QR(self.U,self.R,newSol)

        Uh=self.U.conj().T

        MatsR = []
        for i in range(len(self.Mats)):
            MatsR.append(Uh @ self.Mats[i] @ self.U)

        for i in range(len(self.ports)):
            self.ports[i].setU(self.U)
            # start=timeit.default_timer()
            self.ports[i].create_reduced_modeMats()
            # self.add_time('port_assemble1', start)

        jl.Parallel(n_jobs=4, prefer="threads")(jl.delayed(self.MOR_loop)(MatsR, Uh, fInd) for fInd in range(len(self.fAxis)))
        self.resTot = nlina.norm(self.res_ROM)


    def MOR_loop(self, MatsR, Uh, fInd):
        f = self.fAxis[fInd]

        # start = timeit.default_timer()
        for i in range(len(self.ports)):
            # start = timeit.default_timer()
            if i == 0:
                portMat = self.ports[i].getReducedPortMat(fInd)
            else:
                portMat += self.ports[i].getReducedPortMat(fInd)
        # self.add_time('port_assemble2', start)
        # end port matrix creation

        # start = timeit.default_timer()
        for i in range(len(self.Mats)):
            if i == 0:
                AROM = MatsR[i] * self.factors[i](f)
            else:
                AROM += MatsR[i] * self.factors[i](f)
        AROM += portMat
        # self.add_time('mat_assemble', start)


        # start=timeit.default_timer()
        # self.add_time('projectR', start)

        # start = timeit.default_timer()
        rhs_red = Uh @ self.RHS[:, fInd]
        AQ, AR = slina.qr(AROM)
        vMor = slina.solve_triangular(AR, AQ.conj().T @ rhs_red)
        # self.add_time('solve1', start)

        # start = timeit.default_timer()
        uROM = (self.U @ vMor)  # [:,0]
        # self.add_time('solve2', start)

        # start = timeit.default_timer()
        if fInd in self.fIndsTest:
            i = np.where(self.fIndsTest == fInd)[0][0]
            self.err_R_F[i] = nlina.norm(uROM - self.SolsTest[:, i])
        # self.add_time('err', start)

        self.residual_estimation(fInd, uROM)

        # postprocess solution: impedance+sParams
        # start = timeit.default_timer()
        self.Z[fInd] = impedance(uROM, self.JSrc[:, fInd])
        # self.add_time('Z', start)
        # ZFM[fInd]=impedance(SolsOn[:,fInd],JSrcOn[:,fInd])
        
    def set_residual_indices(self,residual_indices):
        self.residual_indices=residual_indices
        self.Mats_res=[]
        for i in range(len(self.Mats)):
            self.Mats_res.append(self.Mats[i][residual_indices,:])
        tempMat=np.sum(self.Mats_res)
        self.inds_u_res=np.unique(tempMat.indices)
        for i in range(len(self.Mats)):
            self.Mats_res[i]=self.Mats_res[i][:,self.inds_u_res]

    def residual_estimation(self, fInd, uROM):
        f = self.fAxis[fInd]
        # start = timeit.default_timer()
        uROM_short= uROM[self.inds_u_res]
        for i in range(len(self.Mats)):
            if i == 0:
                RHSrom = self.Mats_res[i] @ (self.factors[i](f) * uROM_short)
            else:
                RHSrom += self.Mats_res[i] @ (self.factors[i](f) * uROM_short)
        # self.add_time('resMats', start)

        # start = timeit.default_timer()
        for i in range(len(self.ports)):
            RHSrom += self.ports[i].multiplyVecPortMat(uROM,fInd)[self.residual_indices]

        # self.add_time('resPort', start)
        self.res_ROM[fInd] = nlina.norm(self.RHS[self.residual_indices, fInd] - RHSrom)
        # self.add_time('resPort', start)

    def add_time(self, timeName, start):
        self.times_temp[timeName]+=(timeit.default_timer() - start)


    def select_new_freq(self):
        try:
            selectIndAdaptive(self.nMOR,self.res_ROM,self.solInds)#)f_data['nmor'], res_ROM, solInds)
        except Exception:
            raise Exception('No more frequencies to select')
        return self.fAxis[self.solInds[-1]]

    def get_conv(self):
        resTot=np.linalg.norm(self.res_ROM)
        errTot=np.linalg.norm(self.err_R_F)
        return resTot,errTot,self.res_ROM,self.err_R_F,

    def get_Z(self):
        return self.Z



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    