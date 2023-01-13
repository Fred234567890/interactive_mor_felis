# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import misc

import numpy as np
import oct2py
import os

oc = oct2py.Oct2Py()

path=dict();
path['workDir']=os.path.abspath('').split('\\pythonTests')[0]+'\\pythonTests\\'
path['mats']=path['workDir']+'mats\\'
path['A']=path['mats']+'A\\'
path['ports']=path['mats']+'ports\\'
path['SIBc']=path['mats']+'SIBc\\'
path['RHS']=path['mats']+'RHS\\'


###############################################################################
#read matrices
fAxis=misc.csvRead(path['mats']+'fAxis.csv')

CC=misc.pRead(oc,path['mats']+'CC');
MC=misc.pRead(oc,path['mats']+'MC');
ME=misc.pRead(oc,path['mats']+'ME');
Sols=misc.pRead(oc,path['mats']+'sols');

N=len(fAxis)
rhss,As,SIBcs=(list(),list(),list())
for i in range(N):
    rhss .append(misc.pRead(oc,path['RHS']+'RHS'+str(i)))
    As   .append(misc.pRead(oc,path['A']+'A'+str(i)))
    # SIBcs.append(misc.pRead(oc,path['SIBc']+'SIBc'+str(i)))
    






















