# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:39:17 2022

@author: Fred
"""
import plotfuns

eps  = 8.8541878e-12 
mu = 1.2566371e-06 
c0=-1 
    

import utility as ut
import plotfuns

import numpy as np
import numpy.linalg as nlina
import scipy.linalg as slina
import scipy.sparse.linalg as sslina
import oct2py
import os
import joblib as jl
import timeit

oc = oct2py.Oct2Py()
pRead=lambda path:ut.pRead(oc, path)


path=dict();
path['workDir'] = os.path.abspath('').split('\\interactive_mor_felis')[0]+'\\interactive_mor_felis\\'
path['plots']   = path['workDir']+'_Documentation\\images\\'
path['mats']    = path['workDir']+'mats\\'
path['ports']   = path['mats']+'ports\\'
path['sols']    = path['mats']+'sols\\'
path['sysmats']    = path['mats']+'systemMats\\'






from multiprocessing import Pool, TimeoutError
import time
import os


def f(i,arr1,arr2):
    arr1[i]=i*np.linalg.norm(np.outer(arr2,arr2))

arr1=np.zeros(32)
arr2=np.ones(1000)

times=[]
for i in range(8):
    times.append([])
    numThreads=i+1
    for i in range(10):
        if i==0:
            start=timeit.default_timer()
            for j in range(len(arr1)):
                f(j,arr1,arr2)
            times[-1].append(timeit.default_timer()-start)
        else:
            start=timeit.default_timer()
            jl.Parallel(n_jobs=numThreads, prefer="threads")(jl.delayed(f)(j,arr1,arr2) for j in range(len(arr1)))
            times[-1].append(timeit.default_timer()-start)

times=np.array(times)

fig=plotfuns.initPlot(logY=False,logX=False)
times_mean=[]
for i in range(np.shape(times)[0]):
    times_i=times[i]
    plotfuns.plotCloud(fig,np.ones(len(times_i))*(i+1),times_i,cloudArgs={'color':i,'name':'','showLegend':True})
    times_mean.append(np.mean(times_i))
plotfuns.plotLine(fig,np.array(range(np.shape(times)[0]))+1,times_mean,lineArgs={'color':'black','name':'mean','showLegend':True})
plotfuns.showPlot(fig)























