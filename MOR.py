# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:16:01 2022

@author: Fred
"""

import numpy as np
import numpy.linalg as nlina
import scipy.linalg as slina
import scipy.sparse.linalg as sslina
import subprocess
import zmq
import port
import utility as ut
import warnings
import timeit

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



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    