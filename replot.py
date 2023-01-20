



import utility as ut
import numpy as np
import os
os.environ["PATH"] = os.environ["PATH"] + ";D:\\Ordnerordner\\Software\\pythonEnvironments\\python3_10\\lib\\site-packages\\kaleido\\executable\\"
import plotfuns

path=dict()
path['workDir'] = os.path.abspath('').split('\\interactive_mor_felis')[0]+'\\interactive_mor_felis\\'
path['plots']   = path['workDir']+'_Documentation\\images\\'


def plotConvergence(fig,fileName):
    optsdict={
        'legendShow': True,
        'legendPos': 'topRight',
        'yTick': 3,
        'yFormat': '~e',
        'tickvalsX': [1,5,10,15]#[1,2,5,10,20,50,100,200,500,1000]
             }
    fig.update_xaxes(type='linear')
    plotfuns.exportPlot(fig, fileName, 'half', path=path['plots'], opts=optsdict)
    fig.show()

def plotImpedance(fig,fileName):
    optsdict={
        'legendShow': True,
        'legendPos': 'custom',
        'yTick': 1
             }
    fig.update_xaxes(title_text='Frequency in Hz')
    fig.update_yaxes(title_text='Z in Ohm')
    plotfuns.exportPlot(fig, fileName, 'full', path=path['plots'],opts=optsdict)


fileName='CubeWire_conv1'
fig=plotfuns.loadPlot(fileName,path=path['plots'])
# plotImpedance(fig,fileName)
plotConvergence(fig,fileName)






























