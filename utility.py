# -*- coding: utf-8 -*-
"""
simple wrappers and utility functions to make build in functions better usable
Result file Reading
"""
import os
import sys
import shutil
import warnings

import cmath
import numpy as np
import numpy.linalg as nlina

import scipy.signal as ssignal
# import statisticx as stat

import ast
import pickle
import pandas

def logspace(start,end,n):
    return(np.logspace(np.log10(start),np.log10(end),int(n)))


def allNone(arr):
    #returns wether all entries of n array/tuple are none
    return all(val==None for val in arr)


def minmax(A):
    return(np.min(A),np.max(A))


def dim(A):
    return len(np.shape(A))


def sigRound(x, sig=15):
    from math import log10 , floor
    if dim(x)==0:
        return round(x, sig-int(floor(log10(abs(x))))-1)
    elif dim(x)==1:
        return [round(xi, sig-int(floor(log10(abs(xi))))-1) for xi in x]

    
def normCpx(Y,axis=0,normOrd=2):
    #calculates norms sperately for real and imaginary part
    if len(Y)==0:
        raise Exception('Y cant be empty')
    nRe=nlina.norm(np.real(Y),axis=axis,ord=normOrd)
    nIm=nlina.norm(np.imag(Y),axis=axis,ord=normOrd)
    return(nRe+1j*nIm)


def find(x):
    #x: np.array or list of booleans
    #returns indices of True entries 
    y=np.array(range(len(x)))
    z=y[x]
    return z


def closest_values(arr,vals,returnInd=False):
    arr = np.asarray(arr) 
    foundVals=np.zeros(len(vals))
    inds=np.zeros(len(vals)).astype('int')
    for i in range(len(vals)):
        foundVals[i],inds[i]=closest_value(arr, vals[i],returnInd=True)
    if returnInd:
         return(arr[inds],inds)
    else:
        return arr[inds]
  
        
def closest_value(arr, val, returnInd=False): 
    #16.12.21 https://www.entechin.com/find-nearest-value-list-python/
    arr = np.asarray(arr) 
    i = (np.abs(arr - val)).argmin() 
    if returnInd:
        return(arr[i],i)
    else:
        return arr[i]
    
    
def removeVals(A,b0,axis=0,retMap=False):
    #removes all the elements in b from A        
    Arr=np.array(A)     
    b=np.array(b0)
    if axis==1:
        Arr=Arr.T
        b=b.T
        
    # special case if b is empty
    if np.sum(np.shape(b))==0:
        if retMap==False:        
            return(Arr)
        else:     
            return(Arr,tuple())

    
    # b=np.delete(b0,[])
    if dim(Arr)==1:    ##1D case
        indi=np.zeros(np.shape(Arr))
        for i in range(np.size(b)):
            indi=np.logical_or(indi,Arr == b[i])
        Arr=np.delete(Arr, np.where(indi.astype(dtype=bool)))
        
    else:   ##2D case, A and B are standing, i.e. one element in A corresponds to one row in it
        indi=np.zeros((np.shape(Arr)[0]))    
        for i in range(np.shape(b)[0]):
            indj=np.ones((np.shape(Arr)[0]))
            for j in range(np.shape(b)[1]):
                indj=np.logical_and(indj,Arr[:,j] == b[i,j])
            indi=np.logical_or(indi,indj)
        
        Arr=np.delete(Arr, np.where(indi.astype(dtype=bool)),axis=0)
        
    indMap=(np.where(np.logical_not(indi).astype(dtype=bool))[0],
            np.where(indi.astype(dtype=bool))[0])
    
    if axis==1:
        Arr=Arr.T
        
    if retMap==False:        
        return(Arr)
    else:     
        return(Arr,indMap)


def polarCmplx(u):
    # converts z=a+jb to z=abs+j*phi
    if dim(u)==0:
        ut=cmath.polar(u)
        up=ut[0]+1j*ut[1]
    elif dim(u)==1:
        ab=np.abs(u)
        ph=np.unwrap(np.angle(u),discont=5)
        up=ab+1j*ph        
    elif dim(u)==2:
        up=-np.ones(np.shape(u)).astype('complex')
        for i in range(np.shape(u)[0]):
            ab=np.abs(u[i])
            ph=np.unwrap(np.angle(u[i]),discont=5)
            up[i]=ab+1j*ph

            
    return(up)
    

def append(arr, values, axis=None):
    # appends arrays with compatible shapes, but different dimensions
        
    A=np.array(arr)
    if dim(values)==0:
        b=np.array([values])
    else:
        b=np.array(values)

    dDim=dim(A)-dim(b)
    if dDim==0:
         return np.append(A, b, axis)
    elif dDim==1:        
        if axis==0:
            b=np.array([b])
        elif axis==1:
            b=np.array([b]).T
        return np.append(A,b, axis)
    
    elif dDim==2:
         return np.append(A, np.array([[b]]), axis)
    else:
        raise Exception("incompatible Array dimensions") 
    

def backupFile(path,backupPath,copy=True):
    if os.path.exists(backupPath):
        os.remove(backupPath)
    if copy:
        if os.path.isfile(path):
            shutil.copy(path,backupPath)
        else:
            shutil.copytree(path,backupPath)
            
    else:
        os.rename(path,backupPath)
        
 
def removeReadOnly(path):
    if not os.path. exists(path):
        return
    shutil.rmtree(path)
    
    
def centralDerivative(Y):
    Yderivative=np.zeros(np.shape(Y))
    #forward
    Yderivative[:,0:-1]=np.diff(Y)/2
    #backward
    Yderivative[:,1:]+=np.diff(Y)/2
    #adjust first and last value. They are just 1st order derivative
    Yderivative[:,[0,-1]]*=2
    return Yderivative

###############################################################################
###File reading

def csvRead(fileName,delimiter=None,delim_whitespace=False):
    return  pandas.read_csv(fileName,
                            header=None,
                            delimiter=delimiter,
                            delim_whitespace=delim_whitespace).to_numpy()
    
def pRead(octaveInstance,absPath,cmplx=True,fExtension='.pmat'):
    A=octaveInstance.pRead(absPath+fExtension,cmplx)
    return A

def petscRead(filename, iscomplex=True, indices='int64', precision='float64'):
    if not '.pmat' in filename:
        filename=filename+'.pmat'
        print('added .pmat to filename')
        # warnings.warn('File extension .pmat was added to the filename, fix your code')

    import scipy.sparse as ssp
    if indices == 'int32':
        indices = ">i4"
    elif indices == 'int64':
        indices = ">i8"
    else:
        raise ValueError('Unknown indices type')

    if precision == 'float32':
        precision = ">f4"
    elif precision == 'float64':
        precision = ">f8"
    else:
        raise ValueError('Unknown precision type')

    file=open(filename,'rb')
    # data = np.fromfile(filename, dtype='<f4', count=1)

    filetype=np.fromfile(file, dtype=indices, count=1)[0]
    if filetype == 1211214:
        # vector
        size=np.fromfile(file, dtype=indices, count=1)[0]
        if iscomplex:
            data=np.fromfile(file, dtype=precision, count=2*size)
            v = np.empty(size, dtype=np.complex128)
            v.real= data[0::2]
            v.imag= data[1::2]
        else:
            v=np.fromfile(file, dtype=precision, count=size)

    elif  filetype == 1211216:
        # matrix
        m,n,nz=np.fromfile(file, dtype=indices, count=3)  #m=rows, n=cols, nz=nonzeros
        if nz==-1:
            raise Exception('dense matrix not supported')
        nnz=np.fromfile(file, dtype=indices, count=m)   #nonzeros per row


        sum_nz = sum(nnz)
        if not sum_nz == nz:
            raise Exception('No-Nonzeros sum-rowlengths do not match nz: %d, sum_nz: %d', nz, sum_nz)
        # i=np.ones(nz)

        colInds=np.fromfile(file, dtype=indices, count=nz)

        if nz==m*n:
            #matrix is actually dense
            if iscomplex:
                data=np.fromfile(file, dtype=precision, count=2*nz)
                v = np.empty(nz, dtype=np.complex128)
                v.real= data[0::2]
                v.imag= data[1::2]
                del data
            else:
                v=np.fromfile(file, dtype=precision, count=nz)
            v=v.reshape((m,n))
        else:
            #matrix is actually sparse
            if iscomplex:
                precision = ">f8"
                data=np.fromfile(file, dtype=precision, count=2*nz)
                v = np.empty(nz, dtype=np.complex128)
                v.real= data[0::2]
                v.imag= data[1::2]
            else:
                data=np.fromfile(file, dtype=precision, count=nz)
                v=data
            del data
            indptr=np.zeros(m+1, dtype=indices)
            indptr[1:]=np.cumsum(nnz)
            v=ssp.csr_matrix((v,colInds,indptr),shape=(m,n))
    else:
        file.close()
        raise Exception('File type not supported')
        raise Exception('File type %d not supported', filetype)
    file.close()
    if dim(v)==1:
        v=np.array([v]).T
    return v

###############################################################################
###misc

def endSound():
    import winsound
    winsound.Beep(880, 100)
    winsound.Beep(int(880*1.5), 500)
###############################################################################
###unused funs

