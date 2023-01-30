# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:39:17 2022

@author: Fred
"""

eps  = 8.8541878e-12 
mu = 1.2566371e-06 
c0=-1 
    

import utility as ut

import numpy as np
import numpy.linalg as nlina
import scipy.linalg as slina
import scipy.sparse.linalg as sslina
import oct2py
import os
import multiprocessing as mp

oc = oct2py.Oct2Py()
pRead=lambda path:ut.pRead(oc, path)


path=dict();
path['workDir'] = os.path.abspath('').split('\\interactive_mor_felis')[0]+'\\interactive_mor_felis\\'
path['plots']   = path['workDir']+'_Documentation\\images\\'
path['mats']    = path['workDir']+'mats\\'
path['ports']   = path['mats']+'ports\\'
path['sols']    = path['mats']+'sols\\'
path['sysmats']    = path['mats']+'systemMats\\'



A=np.zeros(10)




from multiprocessing import Pool, TimeoutError
import time
import os

def f(x):
    return x*x

if __name__ == '__main__':
    # start 4 worker processes
    with Pool(processes=4) as pool:

        # print "[0, 1, 4,..., 81]"
        print(pool.map(f, range(10)))

        # print same numbers in arbitrary order
        for i in pool.imap_unordered(f, range(10)):
            print(i)
























