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
    
    def __init__ (self,path,matRead,numDict) :
        self.c0      = 1/np.sqrt (self.mu*self.eps)
        self.path    = path
        self.matRead = matRead
        self.numTB   = 0
        self.numTS   = numDict['TS']
        self.numTE   = numDict['TE']
        self.numTM   = numDict['TM']
        self.f=0
        self.factors = np.zeros(self.getNumModes()).astype('complex')*np.NaN


    def readModesTx (matRead,path,Tx,numModes) :
        fieldsTx=list ()
        # N=0
        for i in range (numModes) :
            fieldsTx.append (matRead(path+Tx+'\\'+str (i)).toarray())
            # if i==0: N=len (fieldsTx[0])
        return fieldsTx
    
    
    def readModes(self):
        if self.numTE>0:
            self.cutoffsTE=self.matRead (self.path+'TE').toarray().T.tolist()[0]
        else:
            self.cutoffsTE=[]
        if self.numTM>0:
            self.cutoffsTM=self.matRead (self.path+'TM').toarray().T.tolist()[0]
        else:
            self.cutoffsTM=[]
        
        self.fieldsTB = list() #readModesTx(self,'TB', self.numTB) 
        self.fieldsTS = Port.readModesTx (self.matRead,self.path,'TS',self.numTS)
        self.fieldsTE = Port.readModesTx (self.matRead,self.path,'TE',self.numTE)
        self.fieldsTM = Port.readModesTx (self.matRead,self.path,'TM',self.numTM)

        
 
    def readMaps(self):
        self.M2D2D=self.matRead(self.path+'M2D2D')
        self.M2D3D=self.matRead(self.path+'M2D3D')
        
        
    def setFrequency(self,f):
        self.f=f
        self.factors*=np.NaN
        
    
    def computeFactors(self):
        kappa=lambda f: 2*np.pi*f/self.c0
        
        for modeInd in range(self.getNumModes()):
            mType=self.getModeType(modeInd) 
            if  mType=='TB' or mType=='TS': 
                self.factors[modeInd]=kappa(self.f)
            else:
                gamma2=kappa (self.getCutoff(modeInd)) **2-kappa(self.f)**2
                if gamma2>=0:
                    gamma=np.sqrt (gamma2) 
                else:
                    gamma=1j*np.sqrt (-gamma2) 
                    
                if mType=='TE':
                   self.factors[modeInd]= gamma 
                elif mType=='TM':
                    self.factors[modeInd]= -kappa(self.f)**2/gamma 
                 
    
    def getFactorOld(self,ind,f):
         kappa=lambda f: 2*np.pi*f/self.c0
         mType=self.getModeType(ind) 
         if  mType=='TB' or mType=='TS': 
             return kappa(f)
         
         den=kappa (self.getCutoff(ind)) **2-kappa(f)**2
         if den>=0:
             gamma=np.sqrt (den) 
         else:
             gamma=1j*np.sqrt (-den) 
             
         if mType=='TE':
            return gamma 
         elif mType=='TM':
             return -kappa(f)**2/gamma 
         
            
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
        
    
    def getModeVec(self,ind):
        return (self.fieldsTB+self.fieldsTS+self.fieldsTE+self.fieldsTM)[ind]
    
    
    def getModeMat(self,ind):
        return self.getModeVec (ind) *self.getModeVec (ind) .T  
        
    
    def getPortMat(self,f):
        A = self.factors[0] *self.getModeMat (0)
        for ind in range (1,self.getNumModes()) :
            A+= self.factors[ind] *self.getModeMat (ind)
        return A

    
    def getReducedModeMat(self,ind,Uh):
        #python: reduce runtime selecting the sice of the NNZ entries out of U in superior fun
        #c++:    reduce runtime by only iterating over the the elements where modeVec is nonzero
        pu=Uh@(self.getModeVec (ind))#.toarray())
        return pu*pu.conj().T  
        
    
    def getReducedPortMat(self,Uh):
        A = self.factors[0] *self.getReducedModeMat (0,Uh)
        for ind in range (1,self.getNumModes()) :
            A+= self.factors[ind] *self.getReducedModeMat (ind,Uh)
        return A
    
    def MultiplyModeMat(self,ind,x):
        fac=self.factors [ind]
        mVec=self.getModeVec (ind)
        return mVec @ (fac * (mVec.T@x) )
        

    def multiplyVecPortMat(self,x):
        y = self.MultiplyModeMat(0,x)
        for ind in range (1,self.getNumModes()) :
            y+= self.MultiplyModeMat(ind,x)
        return y












