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


def getFelisConstants():
    eps  = 8.8541878e-12
    mu = 1.2566371e-06
    c0 = 1/np.sqrt(mu*eps)
    return (eps,mu,c0) #(eps,mu,c0)=MOR.getFelisConstants()

def createMatrices(init_felis,recreate_mats,recreate_test,pRead,path,cond,fmin,fmax,nTrain,nTest,nPorts,nModes):
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
        ports.append(port.Port(path['ports'] + str(i) + '\\', pRead, nModes))
        print('port read modes')
        ports[i].readModes()
        print('port read maps')
        ports[i].readMaps()


    print('read RHS')
    RHS = pRead(path['mats'] + 'RHSs')  #Assumes that RHS is stored as a dense matrix or a 'full' sparse matrix
    print('read Js')
    JSrc = pRead(path['mats'] + 'Js')  #Assumes that JSrc is stored as a dense matrix or a 'full' sparse matrix

    #  create fAxis
    fAxis = np.linspace(fmin, fmax, nTrain)
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

def podOnlineSt2(fAxis,U,ports,Mats,factors,RHSOn,fIndsTest,SolsTest,JSrcOn):

    Uh=U.conj().T
    (res_ROM,err_P_F)=(np.zeros(len(fAxis)),np.zeros(len(fAxis)))
    err_R_F=np.zeros(len(fIndsTest))
    Z=np.zeros(len(fAxis)).astype('complex')
    # ZFM=np.zeros(len(fAxis)).astype('complex')
    #reduce Matrices
    MatsR=[]
    for i in range(len(Mats)):
        MatsR.append(Uh @ Mats[i] @ U)

    for fInd in range(len(fAxis)):
        f=fAxis[fInd]


        #assemble AROM
        for i in range(len(ports)):
            ports[i].setFrequency(f)
            ports[i].computeFactors()
            if i==0: pMatR  = ports[i].getReducedPortMat(Uh)
            else:    pMatR += ports[i].getReducedPortMat(Uh)
        for i in range(len(Mats)):
            if i==0:AROM  = MatsR[i]*factors[i](f)
            else:   AROM += MatsR[i]*factors[i](f)
        AROM += pMatR



        #solve ROM
        vMor = nlina.solve (AROM, Uh @ RHSOn[:,fInd])
        uROM = (U @ vMor) #[:,0]

        if 0:
            #assemble A
            for i in range(len(ports)):
                ports[i].setFrequency(f)
                ports[i].computeFactors()
                if i == 0: pMat = ports[i].getPortMat(Uh)
                else: pMat += ports[i].getPortMat(Uh)
            for i in range(len(Mats)):
                if i == 0: A = Mats[i] * factors[i](f)
                else: A += Mats[i] * factors[i](f)
            A += pMat

            #solve system
            uFull= nlina.solve (A, RHSOn[:,fInd])
            #compute error
            err_P_F[fInd]=np.linalg.norm(uFull-uROM)

        #calculate error
        if fInd in fIndsTest:
            i=np.where(fIndsTest==fInd)[0][0]
            err_R_F[i] = nlina.norm (uROM-SolsTest[:,i])
        
        #calculate residual 
        for i in range(len(Mats)):
            if i==0:RHSrom  = factors[i](f) * (Mats[i] @ uROM)
            else:   RHSrom += factors[i](f) * (Mats[i] @ uROM)
        for i in range(len(ports)):
            RHSrom  += ports[i].multiplyVecPortMat(uROM)
        res_ROM[fInd] = nlina.norm (RHSOn[:,fInd]-RHSrom)
        resTot=nlina.norm(res_ROM)
        
        #postprocess solution: impedance+sParams
        Z[fInd]=impedance(uROM,JSrcOn[:,fInd])
        # ZFM[fInd]=impedance(SolsOn[:,fInd],JSrcOn[:,fInd])
        
    return (resTot,res_ROM,err_R_F,Z,uROM)


# def nested_Gram_Schmidt(U,a):
#     for i in range(np.shape(U)[1]):
#         a -= np.dot(U[:,i],a)*U[:,i]
#         a*=1/np.linalg.norm(a)
#     return a

def nested_QR(U,R,a):
    if U==[]:
        U,R=slina.qr(a,mode='economic')
    else:
        U,R=slina.qr_insert(U,R,a,np.shape(U)[1],which='col')
    return (U,R)

class Pod_adaptive:
    def __init__(self,fAxis,ports,Mats,factors,RHS,JSrc,fIndsTest,SolsTest,nMOR):
        self.fAxis=fAxis
        self.ports=ports
        self.Mats=Mats
        self.factors=factors
        self.RHS=RHS
        self.fIndsTest=fIndsTest
        self.SolsTest=SolsTest
        self.JSrc=JSrc
        self.U=[]
        self.R=[]
        self.res_ROM=np.zeros(len(fAxis))
        self.err_R_F=np.zeros(len(fIndsTest))
        self.Z=np.zeros(len(fAxis)).astype('complex')
        self.uROM=np.zeros(len(fAxis)).astype('complex')
        self.nMOR=nMOR
        self.solInds=[]
        self.sols=[]

        self.MatsR=[]
        self.pMatsR=[]
        for i in range(len(fAxis)):
            self.MatsR.append([])
            self.pMatsR.append([])


        self.times={
            # 'time_mat_assemble_preloop':[0],
            'mat_assemble':[],
            'port_assemble1':[],
            'port_assemble2':[],
            'solve1':[],
            'solve2':[],
            'err':[],
            'resPort':[],
            'resMats':[],
            'Z':[],
        }
        self.times_temp=self.times.copy()




    def update_Classic(self,newSol):
        if self.sols==[]:
            self.sols=newSol
        else:
            self.sols=np.append(self.sols, newSol, axis=1)
        self.U,_,_=slina.svd(self.sols,full_matrices=False)
        (_,res_ROM,err_R_F,Z,uROM)=podOnlineSt2(self.fAxis,self.U,self.ports,self.Mats,self.factors,self.RHS,self.fIndsTest,self.SolsTest,self.JSrc)
        self.res_ROM=res_ROM
        self.err_R_F=err_R_F
        self.Z=Z
        self.uROM=uROM



    def update_nested(self,newSol):

        self.reset_times_temp()

        if self.sols==[]:
            self.sols=newSol
        else:
            self.sols=np.append(self.sols, newSol, axis=1)

        self.U,self.R=nested_QR(self.U,self.R,newSol)

        Uh=self.U.conj().T

        MatsR = []
        for i in range(len(self.Mats)):  #todo nested not exploited
            MatsR.append(Uh @ self.Mats[i] @ self.U)

        for i in range(len(self.ports)): #has to be done only once, not for every frequency
            self.ports[i].setU(self.U)

        for fInd in range(len(self.fAxis)):
            f = self.fAxis[fInd]

            #Port matrix creation:
            for i in range(len(self.ports)):
                start=timeit.default_timer()
                self.ports[i].setFrequency(f)
                self.ports[i].computeFactors()
                self.add_time('port_assemble1', start)
                start = timeit.default_timer()
                if i == 0:
                    new_vals = self.ports[i].getReducedPortMat_nested(self.U)
                else:
                    new_vals += self.ports[i].getReducedPortMat_nested(self.U)  #if only one basis function is used, a matrix is returned, otherwise a vector
                self.add_time('port_assemble2', start)


            if np.shape(self.pMatsR[fInd]) == (0,):
                self.pMatsR[fInd] = np.array([[new_vals[0]]])
            else:
                n=np.shape(self.pMatsR[fInd])[0]+1
                pNew = np.zeros((n,n)).astype('complex')
                pNew[:-1, :-1] = self.pMatsR[fInd]
                pNew[:, -1] = new_vals[n:]
                pNew[-1, :] = new_vals[:n]
                self.pMatsR[fInd] = pNew

            if 0:
                #check port matrix creation
                for i in range(len(self.ports)):
                    self.ports[i].setFrequency(f)
                    self.ports[i].computeFactors()
                    if i == 0:
                        pMatRef = self.ports[i].getReducedPortMat(Uh)
                    else:
                        pMatRef += self.ports[i].getReducedPortMat(Uh)
                print('size basis (%d), find (%d), res portmats: %f' %(np.shape(self.U)[1],fInd,nlina.norm(self.pMatsR[fInd]-pMatRef)))
            #end port matrix creation

            start=timeit.default_timer()
            #exploiting nested structure not necessary here:
            for i in range(len(self.Mats)):
                if i == 0:
                    AROM = MatsR[i] * self.factors[i](f)
                else:
                    AROM += MatsR[i] * self.factors[i](f)
            AROM += self.pMatsR[fInd]
            self.add_time('mat_assemble', start)

            #end exploiting nested structure not necessary here


            start=timeit.default_timer()
            vMor = nlina.solve(AROM, Uh @ self.RHS[:, fInd])
            self.add_time('solve1', start)

            start=timeit.default_timer()
            uROM = (self.U @ vMor)  # [:,0]
            self.add_time('solve2', start)

            start=timeit.default_timer()
            if fInd in self.fIndsTest:
                i = np.where(self.fIndsTest == fInd)[0][0]
                self.err_R_F[i] = nlina.norm(uROM - self.SolsTest[:, i])
            self.add_time('err', start)

            self.residual_estimation(fInd, uROM)


            # postprocess solution: impedance+sParams
            start=timeit.default_timer()
            self.Z[fInd] = impedance(uROM, self.JSrc[:, fInd])
            self.add_time('Z', start)
            # ZFM[fInd]=impedance(SolsOn[:,fInd],JSrcOn[:,fInd])
            #end todo nested not exploited
        self.save_times()
        
        
    def set_residual_indices(self,residual_indices):
        self.residual_indices=residual_indices
        self.Mats_res=[]
        for i in range(len(self.Mats)):
            self.Mats_res.append(self.Mats[i][residual_indices,:])

    def residual_estimation(self, fInd, uROM):
        f = self.fAxis[fInd]
        start = timeit.default_timer()
        for i in range(len(self.Mats)):
            if i == 0:
                RHSrom = self.Mats[i] @ (self.factors[i](f) * uROM)
            else:
                RHSrom += self.Mats[i] @ (self.factors[i](f) * uROM)
        self.add_time('resMats', start)

        start = timeit.default_timer()
        for i in range(len(self.ports)):
            RHSrom += self.ports[i].multiplyVecPortMat(uROM)

        self.res_ROM[fInd] = nlina.norm(self.RHS[:, fInd] - RHSrom)
        self.resTot = nlina.norm(self.res_ROM)
        self.add_time('resPort', start)

    def residual_estimation_orig(self, fInd, uROM):
        f=self.fAxis[fInd]
        start=timeit.default_timer()
        for i in range(len(self.Mats)):
            if i == 0:
                RHSrom = self.Mats[i] @ (self.factors[i](f)*uROM)
            else:
                RHSrom += self.Mats[i] @ (self.factors[i](f)*uROM)
        self.add_time('resMats', start)

        start=timeit.default_timer()
        for i in range(len(self.ports)):
            RHSrom += self.ports[i].multiplyVecPortMat(uROM)
            
        self.res_ROM[fInd] = nlina.norm(self.RHS[:, fInd] - RHSrom)
        self.resTot = nlina.norm(self.res_ROM)
        self.add_time('resPort', start)



    def reset_times_temp(self):
        for i in range(len(self.times_temp)):
            self.times_temp[list(self.times_temp.keys())[i]]=0

    def add_time(self, timeName, start):
        self.times_temp[timeName]+=(timeit.default_timer() - start)

    def save_times(self):
        for i in range(len(self.times_temp)):
            self.times[list(self.times_temp.keys())[i]].append(self.times_temp[list(self.times_temp.keys())[i]])

        # self.times[timeName].append(timeit.default_timer() - start)


    def update_Nested(self,newSol):
        self.U,self.R=nested_QR(self.U,self.R,newSol)

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

    def print_time(self):
        for i in range(len(self.times)):
            print('time_%s: %f' %(list(self.times.keys())[i],np.sum(list(self.times.values())[i])))

    def get_time_for_plot(self):
        #returns the times in a format for plotting
        times=[]
        names=[]
        for i in range(len(self.times)):
            if len(list(self.times.values())[i])>1:
                times.append([])
                names.append(list(self.times.keys())[i])
                for j in range(len(list(self.times.values())[i])):
                    times[i].append(list(self.times.values())[i][j])
        times=np.array(times)
        inds=np.cumsum(np.ones(np.shape(times)[1]))
        return inds,times,names #=get_time_for_plot()

def nested_pod(fAxis,U,ports,Mats,factors,RHSOn,fIndsTest,SolsTest,JSrcOn,matsR):
    Uh = U.conj().T
    (res_ROM, err_P_F) = (np.zeros(len(fAxis)), np.zeros(len(fAxis)))
    err_R_F = np.zeros(len(fIndsTest))
    Z = np.zeros(len(fAxis)).astype('complex')
    # ZFM=np.zeros(len(fAxis)).astype('complex')
    # reduce Matrices
    MatsR = []
    for i in range(len(Mats)):
        MatsR.append(Uh @ Mats[i] @ U)

    for fInd in range(len(fAxis)):
        f = fAxis[fInd]

        # assemble AROM
        for i in range(len(ports)):
            ports[i].setFrequency(f)
            ports[i].computeFactors()
            if i == 0:
                pMatR = ports[i].getReducedPortMat(Uh)
            else:
                pMatR += ports[i].getReducedPortMat(Uh)
        for i in range(len(Mats)):
            if i == 0:
                AROM = MatsR[i] * factors[i](f)
            else:
                AROM += MatsR[i] * factors[i](f)
        AROM += pMatR

        # solve ROM
        vMor = nlina.solve(AROM, Uh @ RHSOn[:, fInd])
        uROM = (U @ vMor)  # [:,0]

        # calculate error
        if fInd in fIndsTest:
            i = np.where(fIndsTest == fInd)[0][0]
            err_R_F[i] = nlina.norm(uROM - SolsTest[:, i])

        # calculate residual
        for i in range(len(Mats)):
            if i == 0:
                RHSrom = factors[i](f) * (Mats[i] @ uROM)
            else:
                RHSrom += factors[i](f) * (Mats[i] @ uROM)
        for i in range(len(ports)):
            RHSrom += ports[i].multiplyVecPortMat(uROM)
        res_ROM[fInd] = nlina.norm(RHSOn[:, fInd] - RHSrom)
        resTot = nlina.norm(res_ROM)

        # postprocess solution: impedance+sParams
        Z[fInd] = impedance(uROM, JSrcOn[:, fInd])
        # ZFM[fInd]=impedance(SolsOn[:,fInd],JSrcOn[:,fInd])

    return (resTot, res_ROM, err_R_F, Z, uROM)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    