def roundLossyWG_config():
    #######MOR CONFIG
    nMax = 10
    f_data= {'fmin' : 0.1e9,
             'fmax' : 0.1e9,
             'nmor' : 1,
             'ntest': 1,          # 1577.186820
             }

    frac_Greedy=1
    # frac_Sparse=1
    n_MC=3570
    accuracy=1e-6
    NChecks=1
    saveMemory=False

    plotConv=False
    plotTime=False
    plotZls=['a'] #'a','r','i',
    plotZts=[] #'a','r','i',

    exportZ=False
    dbName='test'
    orderHex=0
    orderTet=0
    runId=1


    felis_todos=dict()
    felis_todos['init'] =True #dont change

    felis_todos['mats'] =True
    felis_todos['exci'] =True
    felis_todos['test'] =True
    felis_todos['train']=True

    #########MODEL CONFIG
    symmetry=1

    nPorts=2
    nModes={'TS':0,
            'TE':1,
            'TM':0,}

    # cond=1.4e6 #sh12
    # cond=5.8e+3 #cubewire

    return nMax,NChecks,n_MC,frac_Greedy,accuracy,f_data,symmetry,nPorts,exportZ,nModes,orderHex,orderTet,runId,felis_todos,dbName,saveMemory,plotConv,plotTime,plotZls,plotZts



########################################################################################################################
###OLD confifs

def sqc2b_config():
    #######MOR CONFIG
    nMax = 40
    f_data= {'fmin' : 3e9,
             'fmax' : 10e9,
             'nmor' : 20000,
             'ntest': 10,          # 1577.186820
             }

    frac_Greedy=32
    # frac_Sparse=1
    n_MC=174
    accuracy=1e-6
    NChecks=3
    doFiltering=True
    saveMemory=False
    useMC=False

    plotConv=True
    plotTime=False
    plotZls=['a'] #'a','r','i',
    plotZts=['a'] #'a','r','i',

    exportZ=False
    dbName='test'
    orderHex=0
    orderTet=0
    runId=1


    felis_todos=dict()
    felis_todos['init'] =True #dont change

    felis_todos['mats'] =False
    felis_todos['exci'] =False
    felis_todos['test'] =False
    felis_todos['train']=False

    #########MODEL CONFIG
    symmetry=1

    nPorts=2
    nModes={'TB':0,
            'TS':0,
            'TE':5,
            'TM':5,}

    # cond=1.4e6 #sh12
    # cond=5.8e+3 #cubewire
    cond=5.8e+5
    return nMax,NChecks,n_MC,frac_Greedy,accuracy,f_data,useMC,symmetry,nPorts,exportZ,nModes,cond,orderHex,orderTet,runId,felis_todos,dbName,doFiltering,saveMemory,plotConv,plotTime,plotZls,plotZts


##################################################################################################################################
def sh12_3b_config():
    #######MOR CONFIG
    nMax = 40
    f_data= {'fmin' : 80e6,
             'fmax' : 150e6,
             'nmor' : 500,
             'ntest': 10,          # 1577.186820
             }

    frac_Greedy=4
    # frac_Sparse=1
    n_MC=1000
    accuracy=1e-6
    NChecks=3
    doFiltering=True
    saveMemory=False
    useMC=False

    plotConv=True
    plotTime=False
    plotZls=['a','r','i'] #'a','r','i'
    plotZts=['a','r','i'] #'a','r','i'

    exportZ=False
    dbName='test'
    orderHex=0
    orderTet=0
    runId=1


    felis_todos=dict()
    felis_todos['init'] =True #dont change

    felis_todos['mats'] =True
    felis_todos['exci'] =True
    felis_todos['test'] =True
    felis_todos['train']=True

    #########MODEL CONFIG
    symmetry=2

    nPorts=2
    nModes={'TB':0,
            'TS':0,
            'TE':0,
            'TM':1,}

    # cond=1.4e6 #sh12
    # cond=5.8e+3 #cubewire
    cond=1.4e6
    return nMax,NChecks,n_MC,frac_Greedy,accuracy,f_data,useMC,symmetry,nPorts,exportZ,nModes,cond,orderHex,orderTet,runId,felis_todos,dbName,doFiltering,saveMemory,plotConv,plotTime,plotZls,plotZts

