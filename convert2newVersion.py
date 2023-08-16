import numpy as np
import os

from myLibs import utility as ut


def upd_0_9_to_1_0_0():###Update from 0.9 to 1.0.0
    def writeCSV(fileName,value):
        with open(fileName, 'w') as f:
            f.write('  %s,  0.0,  0.0\n' %str(value/1e9))
    #add frequency files for each solution file. The frequency-axis files in RHS and sols are required
    path=dict()
    path['felis_projects']     = os.path.abspath('').split('\\FELIS_Projects')[0]+'\\FELIS_Projects\\'
    path['workDir'] = os.path.abspath('')+'\\'
    path['felis_bin']= path['felis_projects']+"FELIS_Binary\\"
    path['mats']    = path['felis_bin']+'mats\\'
    path['sols']    = path['mats']+'sols\\'

    #take from fAxis in RHS folder
    fMin=50e6
    fMax=2e9
    nf=40e3

    #count in sols folder
    nTest=50

    fAxis=np.linspace(fMin,fMax,int(nf))
    fAxisTest,fIndsTest = ut.closest_values(fAxis, np.linspace(fMin,fMax,nTest),returnInd=True)  #This may be inaccurate if the data was created by multiple runs with different frequency ranges

    fSOls=ut.csvRead(path['sols'] + 'freqs.csv',delimiter=',')[:,0]

    os.mkdir(path['sols'] + 'temp_freqs\\')
    for i in range(nTest):
        writeCSV(path['sols'] + 'temp_freqs\\test_%s.csv'%str(i), fAxisTest[i])
    for i in range(len(fSOls)):
        writeCSV(path['sols'] + 'temp_freqs\\train_%s.csv'%str(i), fSOls[i])
    ###End Update from 0.9 to 1.0.0

upd_0_9_to_1_0_0()