# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import numpy as np
import os

import utility as ut
import plotfuns




path=dict();
path['felis']     = os.path.abspath('').split('\\FELIS')[0]+'\\FELIS\\'
path['workDir']   = os.path.abspath('').split('\\interactive_mor_felis')[0]+'\\interactive_mor_felis\\'
path['impedance'] = '' #path['felis']+'Data\\Beam\\'
path['plots']     = path['workDir']+'_Documentation\\images\\'


plotAll=False
plotMag=True


impedance=ut.csvRead(path['impedance']+'impedance.txt',delim_whitespace=True)


fig=plotfuns.initPlot(title='impedance',logX=False,logY=False,xName='f',yName='Ohm')
plotfuns.plotLines (fig, impedance[:,0]*1e9, impedance[:,1:].T,curveName=['re','im','abs'])

# plotfuns.plotLine (fig, fAxisOn, np.real(ZRef) ,lineArgs={'name':'Py_Re','color':3} )
# plotfuns.plotLine (fig, fAxisOn, np.imag(ZRef) ,lineArgs={'name':'Py_Im','color':4} )

plotfuns.showPlot(fig,show=plotAll)
# plotfuns.exportPlot(fig, 'const_imp_Felis_Py', 'full', path=path['plots'],opts={'legendShow':True})


fig=plotfuns.initPlot(title='impedance',logX=False,logY=True,xName='f',yName='Ohm')
plotfuns.plotLines (fig, impedance[:,0], [impedance[:,3].T])
plotfuns.showPlot(fig,show=plotMag)
print('a')




























