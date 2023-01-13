# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:16:01 2022

@author: Fred
"""

import numpy as np
import numpy.linalg as nlina
import scipy.linalg as slina
import scipy.sparse.linalg as sslina


def impedance(u,j):
    return -np.dot(u,j.conj())

def podOnlineSt2(fAxis,U,ports,Mats,factors,RHSOn,SolsOn,JSrcOn):

    Uh=U.conj().T
    (err_R_F,res_ROM,err_P_F)=(np.zeros(len(fAxis)),np.zeros(len(fAxis)),np.zeros(len(fAxis)))
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
        err_R_F[fInd] = nlina.norm (uROM-SolsOn[:,fInd])
        
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



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    