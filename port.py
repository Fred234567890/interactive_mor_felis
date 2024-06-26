# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:24:06 2022

@author: quetscher
"""
import numpy as np
import scipy as sc
import scipy.sparse.linalg as sslina


class Port:
    eps  = 8.8541878e-12 
    mu = 1.2566371e-06 
    c0=-1 
    
    def __init__ (self,path,matRead,numDict,fAxis) :
        self.c0      = 1/np.sqrt (self.mu*self.eps)
        self.path    = path
        self.matRead = matRead
        self.numTB   = 0
        self.numTS   = numDict['TS']
        self.numTE   = numDict['TE']
        self.numTM   = numDict['TM']
        self.fAxis=fAxis
        self.factors = np.zeros((self.getNumModes(),len(fAxis))).astype('complex')*np.NaN


    def readModesTx (self,matRead,path,Tx,numModes) :
        fieldsTx=list ()
        # N=0
        n = len(self.indices3D)
        col_indices = np.zeros(n, dtype=int)
        for i in range (numModes) :
            newField=matRead(path+Tx+'\\'+str (i))[self.indices3D]
            fieldsTx.append (sc.sparse.csc_matrix((newField[:,0], (self.indices3D, col_indices)), shape=(np.shape(self.M2D3D)[1], 1)))
            # if i==0: N=len (fieldsTx[0])
        return fieldsTx


    def readModes(self):
        if self.numTE>0:
            self.cutoffsTE=np.real(self.matRead (self.path+'TE')).T.tolist()[0]
        else:
            self.cutoffsTE=[]
        if self.numTM>0:
            self.cutoffsTM=np.real(self.matRead (self.path+'TM')).T.tolist()[0]
        else:
            self.cutoffsTM=[]
        
        self.fields = list() #readModesTx(self,'TB', self.numTB)
        self.fields+=self.readModesTx (self.matRead,self.path,'TS',self.numTS)
        self.fields+=self.readModesTx (self.matRead,self.path,'TE',self.numTE)
        self.fields+=self.readModesTx (self.matRead,self.path,'TM',self.numTM)

 
    def readMaps(self):
        self.M2D2D=self.matRead(self.path+'M2D2D')
        self.M2D3D=self.matRead(self.path+'M2D3D')
        non_zero_columns = set()
        for i in range(self.M2D3D.shape[0]):
            non_zero_columns.update(self.M2D3D.indices[self.M2D3D.indptr[i]:self.M2D3D.indptr[i + 1]])
        self.indices3D = sorted(list(non_zero_columns))

    # def readEInc(self):
    #     self.EIncs=self.matRead(self.path+'EInc')
    #

    def computeFactors(self):
        kappa=lambda f: 2*np.pi*f/self.c0

        for fInd in range(len(self.fAxis)):
            f=self.fAxis[fInd]
            for modeInd in range(self.getNumModes()):
                mType=self.getModeType(modeInd)
                if  mType=='TB' or mType=='TS':
                    self.factors[modeInd,fInd]=kappa(f)
                else:
                    gamma2=kappa (self.getCutoff(modeInd)) **2-kappa(f)**2
                    if gamma2>=0:
                        gamma=np.sqrt (gamma2)
                    else:
                        gamma=1j*np.sqrt (-gamma2)

                    if mType=='TE':
                       self.factors[modeInd,fInd]= gamma
                    elif mType=='TM':
                        self.factors[modeInd,fInd]= -kappa(f)**2/gamma

    def getFactor(self,ind,fInd): #self.getFactor(ind,fInd)
        return self.factors[ind,fInd]
            
    def getNumModes(self):
        return self.numTB+self.numTS+self.numTE+self.numTM
        
 
    def getCutoff(self,ind):
        fcs =  []
        # fcs += np.zeros(self.numTB+self.numTS).tolist()
        fcs += self.cutoffsTE
        fcs += self.cutoffsTM
        return fcs[ind-(self.numTB+self.numTS)] 
    
    
    def getModeType(self,ind):
        ind=ind-self.numTB 
        if ind<0:
           return 'TB'  
        ind=ind-self.numTS 
        if ind<0:
           return 'TS'  
        ind=ind-self.numTE 
        if ind<0:
           return 'TE'  
        ind=ind-self.numTM 
        if ind<0:
           return 'TM'  
        raise Exception('index to high')

    def create_reduced_modeMats(self):
        self.fields_reduced = list()
        Uh = self.U.T.conj()
        for ind in range(self.getNumModes()):
            pu = Uh @ (self.getModeVec(ind))
            self.fields_reduced.append(np.outer(pu, pu.conj()))

    def getReducedModeMat(self,ind):
        return self.fields_reduced[ind] #todo fields_reduces is a bad name.



    def getReducedPortMat(self, fInd):
        A = self.getFactor(0, fInd) * self.getReducedModeMat(0)
        for ind in range(1, self.getNumModes()):
            A += self.getFactor(ind, fInd) * self.getReducedModeMat(ind)
        return A

    def getModeVec(self,ind):
        return self.fields[ind]
    
    def getModeMat(self,ind):
        return self.getModeVec (ind) *self.getModeVec (ind) .T
    
    def getPortMat(self,fInd):
        A = self.getFactor(0, fInd) *self.getModeMat (0)
        print('Gamma='+str(self.getFactor(0, fInd)))
        print('outer norm='+str(sslina.norm(self.getModeMat (0),'fro')))
        for ind in range (1,self.getNumModes()) :
            A+= self.getFactor(ind, fInd) *self.getModeMat (ind)
            print('outer norm='+str(sslina.norm(self.getModeMat (ind),'fro')))
            print('Gamma='+str(self.getFactor(ind, fInd)))
        return A

    def MultiplyModeMat(self,ind,fInd):
        fac=self.getFactor(ind, fInd)
        mVec=self.getModeVec (ind).data
        return mVec * (fac * (np.dot(mVec.conj(),self.x_short)))

    def multiplyVecPortMatSparse(self,x,fInd):
        self.x_short=x[self.getModeVec(0).indices]
        y=np.zeros(len(self.x_short)).astype('complex')
        for ind in range (self.getNumModes()) :
            y+= self.MultiplyModeMat(ind,fInd)
        return y

    def get3Dinds(self):
        return self.getModeVec(0).indices
    def setU(self,U):
        self.U_short=U[self.getModeVec(0).indices,:]  #select only the rows of U according to the sparsity structure of the modeVecs
        self.U=U

    # def getSParam(self,u,fInd,ind):
    #     mType=self.getModeType(ind)
    #     Einc=self.EIncs[:,fInd]
    #     if mType=='TE':
    #         Etmp=-self.M2D2D @ Einc
    #         Etmp= self.M2D3D @ u + Etmp
    #         np.dot(Etmp,self.getModeVec(ind))
    #
    #
    # def getSParams(self,u,fInd):
    #     for ind in range (self.getNumModes()) :
    #         self.getSParam(u,fInd,ind)








