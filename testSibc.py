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

import port
import MOR

oc = oct2py.Oct2Py()
pRead=lambda path:ut.pRead(oc, path)


path=dict();
path['workDir'] = os.path.abspath('').split('\\interactive_mor_felis')[0]+'\\interactive_mor_felis\\'
path['plots']   = path['workDir']+'_Documentation\\images\\'
path['mats']    = path['workDir']+'mats\\'
path['ports']   = path['mats']+'ports\\'
path['sols']    = path['mats']+'sols\\'
path['sysmats']    = path['mats']+'systemMats\\'

CC = ut.petscRead (path['mats'] + 'CC')
CC2= pRead        (path['mats'] + 'CC')
diffMat=CC-CC2

diff=sslina.norm(CC-CC2)




























