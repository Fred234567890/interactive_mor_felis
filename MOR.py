# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:16:01 2022

@author: Fred
"""

import glob
import struct
import subprocess
import timeit
from pathlib import Path
import os

import joblib as jl
import numpy as np
import numpy.linalg as nlina
import scipy.linalg as slina
import zmq
from myLibs import utility as ut

import misc
import port


class Beam:
    def __init__(self,J,pos,driving):
        self.J=J

        if len(pos)!=2:
            raise Exception('position needs to be tuple/list... of the x and y coordinate')
        if not isinstance(pos[0], float) or not isinstance(pos[1], float):
            raise Exception('x and y coordinates need to be floats')
        self.pos=pos
        self.driving=driving

    def getDofInds(self):
        return self.J[:,0].indices

    def getJ(self):
        return self.J

    def getPosition(self):
        return self.pos
class Beams:
    def __init__(self):
        self.drivingBeam=-1
        self.beams=[]

    def addBeam(self,J,pos,driving):
        self.beams.append(Beam(J,pos,driving))
        if self.drivingBeam!=-1 and driving:
            raise Exception("driving beam already defined")
        if driving:
            self.drivingBeam=len(self.beams)-1

    def getN(self):
        return len(self.beams)

    def getDofInds(self, beamInd=None):
        if beamInd==None:
            inds=[]
            for i in range(self.getN()):
                inds.append(self.beams[i].getDofInds())
            return np.unique(inds)
        else:
            return self.beams[beamInd].getDofInds()

    def getJ(self,iBeam):
        return self.beams[iBeam].getJ()

    def getDrivingBeam(self):
        if self.drivingBeam==-1:
            raise Exception('no driving beam')
        return self.drivingBeam

    def getPosition(self, iBeam):
        return self.beams[iBeam].getPosition()

def getFelisConstants():
    eps  = 8.8541878e-12
    mu = 1.2566371e-06
    c0 = 1/np.sqrt(mu*eps)
    return (eps,mu,c0) #(eps,mu,c0)=MOR.getFelisConstants()

def writeFAxis(fileName,values):
    directory = Path(fileName).parent
    directory.mkdir(parents=True, exist_ok=True)
    with open(fileName, 'w') as f:
        f.write(str(len(values))+'\n')
        for i in range(len(values)):
            f.write(str(values[i])+'\n')


def createMatrices(felis_todos,pRead,path,fmin,fmax,nTrain,nTest,nPorts,nModes):

    #  create fAxis
    fAxis = np.linspace(fmin, fmax, nTrain)

    #  delete old files
    if felis_todos['train']:
        subprocess.call("del /q /s " + path['sols'] + "\\train*", shell=True)
        subprocess.call("del /q /s " + path['sols'] + "\\freqs.csv", shell=True)
    else:
        fExisting = np.sort(ut.csvRead(path['sols'] + 'freqs.csv',delimiter=',')[:,0])
        fAxis= np.sort(np.append(fAxis,fExisting))
        nans=0
        # diff=np.zeros(len(fAxis)-1)
        ref=fAxis[0]
        for i in range(len(fAxis)-1):
            # diff[i]=fAxis[i+1]-fAxis[i]
            if fAxis[i+1]-fAxis[i]<1e-10*ref:
                fAxis[i]=np.NAN
                nans+=1
        if nans>0:
            fAxis=np.sort(fAxis)[0:-nans]

    if not felis_todos['exci']:
        # check if fAxis changed and rhs has to be recalculated
        fAxisChanged=False
        try:
            frhsOld=ut.csvRead(path['rhs'] + 'fAxis.csv',delimiter=',')[1:,0]
            if not np.allclose(frhsOld,fAxis,rtol=1e-10,atol=1e100):
                fAxisChanged=True
        except:
            ...
        if fAxisChanged:
            raise Exception('fAxis changed, redo RHS')

    #  Create socket to talk to server
    context = zmq.Context()
    misc.timeprint("Connecting to Felis")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    misc.timeprint("Connected to Felis")

    if felis_todos['init']:
        misc.timeprint("Initializing Felis")
        socket.send_string("initialize")
        message = socket.recv()
        misc.timeprint(message)

    #recreate matrices
    if felis_todos['mats']:
        subprocess.call("del /q " + path['mats'] + "\\*", shell=True)
        subprocess.call("del /q /s " + path['ports'] + "\\*", shell=True)
        #  export system matrices
        socket.send_string("export_mats")
        message = socket.recv()
        misc.timeprint(message)

    if felis_todos['exci']:
        subprocess.call("del /q " + path['rhs'] + "\\*", shell=True)
        writeFAxis(path['rhs'] + 'fAxis.csv', fAxis)
        # socket.send_string("export_RHSs: %f, %f, %d" % (fmin, fmax, nTrain))
        socket.send_string("export_RHSs")
        message = socket.recv()
        misc.timeprint(message)

    ###############################################################################
    ###import/create constant matrices

    # w = lambda freq: 2 * np.pi * freq
    # kap = lambda freq: w(freq) / c0
    # if useMC:
    #     Mats = [CC, ME, MC, Sibc]
    #     factors = [
    #         lambda f: 1,
    #         lambda f: -kap(f) ** 2,
    #         lambda f: 1j * w(f) * mu,
    #     ]

    w = lambda freq: 2 * np.pi * freq
    Mats = [pRead(path['mats'] + 'CC'),pRead(path['mats'] + 'ME')]
    facs=[lambda f: 1,lambda f:  -(w(f)/getFelisConstants()[2])**2]

    if(os.path.exists(path['mats'] + 'MC')):
        Mats.append(pRead(path['mats'] + 'MC'))
        facs.append(lambda f: 1j * w(f) * mu)

    SibcMats=[];
    #read conductivities and sibcE
    conds=[]
    if os.path.exists(path['mats'] + 'conductivities.csv'):
        if os.path.getsize(path['mats'] + 'conductivities.csv')>0:
            conds=ut.csvRead(path['mats'] + 'conductivities.csv',delimiter=',')[:,0]

    for i in range(len(conds)):
        Mats.append( pRead(path['mats'] + 'Sibc'+str(i)))
        # facs.append(lambda f:-1j*2*np.pi*f*getFelisConstants()[1]/sqrt())
        cond=conds[i]
        fun=lambda f:-1/2*(1+1j)*np.sqrt(2*np.pi*f*getFelisConstants()[1]*cond)
        facs.append(fun)
    # if len(conds)==0:
    #     Mats.append(ME * 0)
    #     facs.append(lambda f:0)

    #facs,Mats,sibcInd


    ports = []
    for i in range(nPorts):
        ports.append(port.Port(path['ports'] + str(i) + '\\', pRead, nModes,fAxis))
        misc.timeprint('port read maps')
        ports[i].readMaps()
        misc.timeprint('port read modes')
        ports[i].readModes()
        misc.timeprint('port compute factors')
        ports[i].computeFactors()
        # misc.timeprint('port read EInc')
        # ports[i].readEInc()



    misc.timeprint('read RHS')
    RHS = pRead(path['rhs'] + 'RHSs').tocsc()  #Assumes that RHS is stored as a dense matrix or a 'full' sparse matrix
    misc.timeprint('read Js')
    Jpath = path['rhs']+'J*.pmat'
    filenames = glob.glob(Jpath)
    beams=Beams()
    for i in range(len(filenames)):
        with open(filenames[i][:-5]+'.info', 'rb') as fInfo:
            driving=struct.unpack('?', fInfo.read(1))[0]
            x = struct.unpack('d', fInfo.read(8))[0]
            y = struct.unpack('d', fInfo.read(8))[0]
            J = pRead(filenames[i][:-5]).tocsc()
            beams.addBeam(J,(x,y),driving)

    fAxisTest,fIndsTest = ut.closest_values(fAxis, np.linspace(fmin,fmax,nTest),returnInd=True)  # select test frequencies from fAxis
    # create test data

    if felis_todos['test']:
        subprocess.call("del /q /s " + path['sols'] + "\\test*", shell=True)

    runtimeTest=0
    Z_felis=np.zeros((beams.getN(),nTest),dtype=complex)
    for i in range(nTest):
        if felis_todos['test']:
            timeStart = timeit.default_timer()
            socket.send_string("solve: test_%d %f" % (i, fAxisTest[i]))
            message = socket.recv()
            runtimeTest+=timeit.default_timer() - timeStart
            misc.timeprint(message)
        if i == 0:
            sols_test = pRead(path['sols'] + 'test_0')
        else:
            sols_test = np.append(sols_test, pRead(path['sols'] + 'test_%d' % i), axis=1)
        Z_data=ut.csvRead(path['sols'] + 'test_%d' % i + '.csv',delimiter=',')[0,1:]
        for j in range(beams.getN()):
            Z_felis[j,i]=(Z_data[2*j+0]+1j*Z_data[2*j+1])
    misc.timeprint('runtime Felis for test data: %f' %runtimeTest)
    return (facs,Mats,ports,RHS,beams,fAxis,fAxisTest,fIndsTest,sols_test,Z_felis,socket)



def impedance(u,j,symmetry):
    return -np.vdot(j,u)/symmetry


def nested_QR(U,R,a):
    if np.shape(U)[1]==0:
        U,R=slina.qr(a,mode='economic')
    else:
        U,R=slina.qr_insert(U,R,a,np.shape(U)[1],which='col')
    return (U,R)

class Pod_adaptive:
    def __init__(self,fAxis,ports,factors,Mats,RHS,beams,symmetry,fIndsTest,SolsTest,nMax):
        self.fAxis=fAxis
        self.fIndsEval=list(range(len(fAxis)))
        self.ports=ports
        self.Mats=Mats
        self.MatsR = []
        self.factors=factors
        self.symmetry=symmetry

        self.RHS=RHS
        self.rhs_inds=np.unique(np.sort(RHS.indices))
        self.RHS_dense=RHS[self.rhs_inds,:].toarray()
        self.rhs_map=([],[]) #first list: indices where to search in RHS_dense, second list: indices where to map to for residual estimation

        self.fIndsTest=fIndsTest
        self.SolsTest=SolsTest
        self.beams=beams
        self.U=np.array([[]])
        self.R=[]
        # self.rhs_reds=np.zeros((nMax,len(fAxis))).astype('complex')
        self.res_ROM=np.zeros(len(fAxis))
        self.err_R_F=np.zeros(len(fIndsTest))
        self.Zl=[np.zeros(len(fAxis)).astype('complex') for i in range(beams.getN())]
        self.Zt=[np.zeros(len(fAxis)).astype('complex') for i in range(beams.getN()-1)]
        self.nMOR=len(fAxis)
        self.solInds=[]
        self.parallel=False
        self.timing=True
        self.times={
            'MOR'         :np.zeros((nMax,len(fAxis))),
            'FEM'           :np.zeros((nMax,len(fAxis))),
            'preparations1' :np.zeros((nMax,len(fAxis))),
            'preparations2' :np.zeros((nMax,len(fAxis))),
            'preparations3' :np.zeros((nMax,len(fAxis))),
            'mat_assemble'  :np.zeros((nMax,len(fAxis))),
            'port_assemble' :np.zeros((nMax,len(fAxis))),
            'project_RHS'   :np.zeros((nMax,len(fAxis))),
            'solve_LGS_QR'  :np.zeros((nMax,len(fAxis))),
            'solve_proj'    :np.zeros((nMax,len(fAxis))),
            'res_port'      :np.zeros((nMax,len(fAxis))),
            'res_mats'      :np.zeros((nMax,len(fAxis))),
            'res_norm'      :np.zeros((nMax,len(fAxis))),
            'Z'             :np.zeros((nMax,len(fAxis))),
            'err'           :np.zeros((nMax,len(fAxis))),
            'misc'          :np.zeros((nMax,len(fAxis))),
        }
        self.resTMP=np.zeros(np.shape(Mats[0])[0]).astype('complex')
        self.uROM=None

    def directSolveAtTest(self,indFTest):
        import scipy.sparse.linalg as sslina
        f=self.fAxis[ self.fIndsTest[indFTest]]
        # operator=self.Mats[0]*self.factors[0](f)
        # for i in range(1,len(self.Mats)):
        #     operator+=self.Mats[i]*self.factors[i](f)
        w = lambda freq: 2 * np.pi * freq
        fac = lambda f: -(w(f) / getFelisConstants()[2]) ** 2

        operator=self.Mats[0]*0
        for i in range(len(self.ports)):
            operator+=self.ports[i].getPortMat(self.fIndsTest[indFTest])
        print('norm Port operator:' + str(sslina.norm(operator, 'fro')))
        operator+=self.Mats[0]+fac(f)*self.Mats[1]


        print('norm operator:' + str(sslina.norm(operator, 'fro')))
        sol=sslina.spsolve(operator,self.RHS[:,self.fIndsTest[indFTest]].toarray())
        print('2norm sol: '+str(nlina.norm(sol)))
        return sol


    def update(self,newSol):
        start = timeit.default_timer()
        if np.shape(self.U)[1]==0:
            self.uROM=newSol[:,0]*0
        self.nBasis=np.shape(self.U)[1]+1
        self.add_time('misc', start,0)

        start = timeit.default_timer()
        self.U,self.R=nested_QR(self.U,self.R,newSol)
        Uh=self.U.conj().T
        Uh_rhs=Uh[:,np.unique(np.sort(self.RHS.indices))]
        self.Ushort=self.U[self.inds_u_proj,:]
        self.uShort=np.zeros(len(self.inds_u_proj)).astype('complex')
        self.add_time('preparations1', start,0)


        start = timeit.default_timer()
        if self.nBasis==1:
            #MatsRtemp = []
            for i in range(len(self.Mats)):
                self.MatsR.append(Uh @ self.Mats[i] @ self.U)
        else:
            for i in range(len(self.Mats)):
                matTemp=np.zeros((self.nBasis,self.nBasis),dtype='complex')
                matTemp[:self.nBasis-1,:self.nBasis-1]=self.MatsR[i]
                self.MatsR[i]=matTemp
                newRow=Uh[-1,:] @ self.Mats[i] @ self.U[:,:self.nBasis]
                self.MatsR[i][:,-1]=newRow.conj()
                self.MatsR[i][-1,:]=newRow
                # MatsRtemp.append(Uh @ self.Mats[i] @ self.U)
        self.add_time('preparations2', start,0)

        start = timeit.default_timer()
        for i in range(len(self.ports)):
            self.ports[i].setU(self.U)
            self.ports[i].create_reduced_modeMats()
        self.add_time('preparations3', start,0)
        n_jobs = 1
        if n_jobs== 1:
            for fInd in range(len(self.fAxis)):
                self.MOR_loop(self.MatsR, Uh, Uh_rhs, fInd)
        else:
            raise Exception('parallel currently disabled')
            jl.Parallel(n_jobs=n_jobs, prefer="threads")(jl.delayed(self.MOR_loop)(MatsR, Uh, fInd) for fInd in range(len(self.fAxis)))
        self.resTot = nlina.norm(self.res_ROM)


    def MOR_loop(self, MatsR, Uh, Uh_rhs, fInd):
        start = timeit.default_timer()
        if not fInd in self.fIndsEval:
            return
        f = self.fAxis[fInd]
        self.add_time('misc', start,fInd)
        start = timeit.default_timer()
        for i in range(len(self.ports)):
            if i == 0:
                portMat = self.ports[i].getReducedPortMat(fInd)
            else:
                portMat += self.ports[i].getReducedPortMat(fInd)
        self.add_time('port_assemble', start,fInd)
        # end port matrix creation

        start = timeit.default_timer()

        AROM = MatsR[0] * self.factors[0](f)
        for i in range(1,len(self.Mats)):
            AROM += MatsR[i] * self.factors[i](f)

        AROM += portMat
        self.add_time('mat_assemble', start,fInd)


        # start=timeit.default_timer()
        # self.add_time('projectR', start)

        # start = timeit.default_timer()
        # rhs_red_old = Uh @ self.getRHS(fInd)
        # self.add_time('project_RHS_1', start,fInd)


        start = timeit.default_timer()
        rhs_red= Uh_rhs @ self.RHS_dense[:,fInd]
        self.add_time('project_RHS', start,fInd)

        # start = timeit.default_timer()
        # rhs_red=self.rhs_reds[0:n+1,fInd]
        # self.add_time('project_RHS_2_2', start,fInd)

        # Alternative solvers
        # start=timeit.default_timer()
        # vMor = slina.solve(AROM, rhs_red)
        # self.add_time('solve_LGS_sci', start,fInd)
        # start = timeit.default_timer()
        # vMor = nlina.solve(AROM, rhs_red)
        # self.add_time('solve_LGS_np', start,fInd)
        # if fInd==0:
        #     self.vMor_old=vMor
        # start = timeit.default_timer()
        # vMor,code = scipy.sparse.linalg.cg(AROM, rhs_red, x0=self.vMor_old)
        # if not code ==0:
        #     print('cg did not converge: '+str(code))
        # self.add_time('solve_LGS_cg', start,fInd)
        # start = timeit.default_timer()
        # vMor = scipy.sparse.linalg.bicg(AROM, rhs_red, x0=self.vMor_old)[0]
        # self.add_time('solve_LGS_bicg', start,fInd)
        # self.vMor_old=vMor

        start = timeit.default_timer()
        # rhs_red = Uh @ self.getRHS(fInd)
        AQ, AR = slina.qr(AROM)
        vMor = slina.solve_triangular(AR, AQ.conj().T @ rhs_red)
        self.add_time('solve_LGS_QR', start,fInd)

        start = timeit.default_timer()
        self.uROM[self.inds_u_proj]=self.Ushort @ vMor
        self.add_time('solve_proj', start,fInd)



        self.residual_estimation(fInd)

        if fInd in self.fIndsTest:
            start = timeit.default_timer()
            i = np.where(self.fIndsTest == fInd)[0][0]
            self.err_R_F[i] = nlina.norm(self.uROM[self.inds_u_proj] - self.SolsTest[self.inds_u_proj, i],ord=2)/nlina.norm(self.SolsTest[self.inds_u_proj, i],ord=2)
            # self.uErr=self.U @ vMor
            # self.err_R_F[i] = nlina.norm(self.uErr - self.SolsTest[:, i],ord=2)/nlina.norm(self.SolsTest[:, i],ord=2)
            self.add_time('err', start,fInd)

        # postprocess solution: impedance
        start = timeit.default_timer()
        for beamInd in range(self.beams.getN()):
            dofInds=self.beams.getDofInds(beamInd)
            self.Zl[beamInd][fInd] = impedance(self.uROM[self.beams.getDofInds(beamInd)], self.beams.getJ(beamInd)[self.beams.getDofInds(beamInd), fInd].toarray(),self.symmetry) #main code -> don't delete
        self.add_time('Z', start,fInd)

        #old:
        # self.Z[fInd] = impedance(self.uROM[self.rhs_inds], self.JSrc[self.rhs_inds, fInd].data,self.symmetry)
        # self.add_time('Z', start,fInd)
        # ZFM[fInd]=impedance(SolsOn[:,fInd],JSrcOn[:,fInd])



    def sParam(self,u,f):
        for i in range(len(self.ports)):
            S=self.ports[i].getSParams(u,f)


    def set_residual_indices(self,residual_indices,frac_Sparse):

        #creating the residual matrices and the indexing for the solution vector
        print('number DoFs considered for res: %d' %(len(residual_indices)))
        # new
        self.residual_indices=residual_indices  #the indices of the solution that are considered for the residual estimation
        self.Mats_res=[]
        for i in range(len(self.Mats)):
            self.Mats_res.append(self.Mats[i][residual_indices,:])  #The row indexed matrices for the residual computation
        tempMat=np.sum(self.Mats_res)
        self.inds_u_res=np.unique(tempMat.indices) #the indices of the solution that are required for the matrix vector product. Generally more indices than residual_indices
        for i in range(len(self.Mats)):
            self.Mats_res[i]=self.Mats_res[i][:,self.inds_u_res] #The fully indexed matrices for the residual computation

        #needed exclusively for residual computation
        self.rhs_map=(-np.ones(len(residual_indices),dtype=int),-np.ones(len(residual_indices),dtype=int))
        for i in range(len(residual_indices)):
            if residual_indices[i] in self.rhs_inds:
                self.rhs_map[0][i]=np.where(self.rhs_inds==residual_indices[i])[0][0]
                self.rhs_map[1][i]=i
        self.rhs_map=(np.unique(self.rhs_map[0])[1:],np.unique(self.rhs_map[1])[1:])
        self.rhs_res=self.RHS_dense[self.rhs_map[0],:]
        self.rhs_normed=(nlina.norm(self.RHS_dense,axis=0,ord=2)**2/frac_Sparse)**0.5  #correction factor because less elements are in the sum
        self.rhs_short = np.zeros(len(residual_indices)).astype('complex')  #initialize a zero vector

    def set_projection_indices(self):
        """
        Set the indices of the projection matrix that are used to project the reduced basis onto the solution. These
        indices constist of the residual indices, of the ports and of the beams.
        Returns
        -------

        """
        inds_u_res=self.inds_u_res
        inds_rhs=self.RHS[:,0].indices
        self.inds_u_proj=np.sort(np.unique(np.concatenate((inds_u_res,inds_rhs,self.beams.getDofInds()))))

    def residual_estimation(self, fInd):
        start = timeit.default_timer()
        f = self.fAxis[fInd]
        uROM_short = self.uROM[self.inds_u_res]
        for i in range(len(self.Mats)):
            if i == 0:
                RHSrom = self.Mats_res[i] @ (self.factors[i](f) * uROM_short)
            else:
                RHSrom += self.Mats_res[i] @ (self.factors[i](f) * uROM_short)
        self.add_time('res_mats', start,fInd)




        start = timeit.default_timer()
        for i in range(len(self.ports)):
            self.resTMP[self.ports[i].get3Dinds()]=self.ports[i].multiplyVecPortMatSparse(self.uROM,fInd)
            RHSrom += self.resTMP[self.residual_indices]
            self.resTMP[self.ports[i].get3Dinds()] *=0
        self.add_time('res_port', start,fInd)



        start = timeit.default_timer()
        self.rhs_short[self.rhs_map[1]] = self.rhs_res[:, fInd]
        self.res_ROM[fInd] = nlina.norm(self.rhs_short - RHSrom,ord=2)/self.rhs_normed[fInd]
        self.add_time('res_norm', start, fInd)


    def add_time(self, timeName, start,fInd):
        if self.timing:
            self.times[timeName][self.nBasis-1,fInd]+=(timeit.default_timer() - start)

    def get_time_for_plot(self):
        timeSum = np.zeros(np.shape(self.times['MOR'])[0])
        times = []
        names = []
        for i in range(len(self.times)):
            times.append([])
            names.append(list(self.times.keys())[i])
            times[-1]=np.sum(self.times[names[i]],axis=1)
            if not names[i] == 'MOR':
                timeSum += times[-1]
        names.append('sum')
        times.append(timeSum)

        inds = np.cumsum(np.ones(np.shape(times)[1]))
        return inds,times,names



    def fAxisGreedy(self,density):
        self.fIndsEval=np.round(np.linspace(0,len(self.fAxis)-1,int(len(self.fAxis)*density))).astype(int)
        self.fIndsEval=np.sort(np.unique(np.append(self.fIndsEval,self.fIndsTest)))


    def fAxisGreedy_treeRefine(self, refine_ind):
        ind_fIndsEval = np.where(self.fIndsEval == refine_ind)[0][0]
        if ind_fIndsEval+1 <= len(self.fIndsEval)-1:
            if self.fIndsEval[ind_fIndsEval+1]-self.fIndsEval[ind_fIndsEval]>1:
                indNew= np.round((self.fIndsEval[ind_fIndsEval]+self.fIndsEval[ind_fIndsEval+1])/2).astype(int)
                self.fIndsEval=np.insert(self.fIndsEval,ind_fIndsEval+1,indNew)
            else:
                misc.timeprint('no refinement possible for higher frequency')
        if ind_fIndsEval-1 >= 0:
            if self.fIndsEval[ind_fIndsEval]-self.fIndsEval[ind_fIndsEval-1]>1:
                indNew= np.round((self.fIndsEval[ind_fIndsEval]+self.fIndsEval[ind_fIndsEval-1])/2).astype(int)
                self.fIndsEval=np.insert(self.fIndsEval,ind_fIndsEval,indNew)
            else:
                misc.timeprint('no refinement possible for lower frequency')

    def select_new_freq_greedy(self):
        fInds_exclusive_test=np.setdiff1d(self.fIndsEval,self.fIndsTest)
        if len(self.solInds) == 0:
            a=int(len(fInds_exclusive_test) / 2)
            newSolInd=fInds_exclusive_test[a]
            self.solInds.append(newSolInd)
        else:
            facts= np.array([0 if i in self.solInds else 1 for i in range(len(self.fAxis))])
            if np.sum(facts)==0:
                raise Exception('no more frequencies available')
            res = (self.res_ROM*facts)[fInds_exclusive_test]
            newSolInd= fInds_exclusive_test[np.argmax(res)]
            self.solInds.append(newSolInd)
            self.fAxisGreedy_treeRefine(newSolInd)
        return self.fAxis[newSolInd]

    def select_new_freq(self):
        if len(self.solInds) == 0:
            self.solInds.append((len(self.fAxis) / 2).astype(int))
        else:
            facts = np.array([0 if i in self.solInds else 1 for i in range(len(self.fAxis))])
            if np.sum(facts) == 0:
                raise Exception('no more frequencies available')
            self.solInds.append(np.argmax(self.res_ROM * facts))
        return self.fAxis[self.solInds[-1]]

    def register_new_freq(self, freq):
        self.solInds.append(np.where(np.abs((self.fAxis - freq))<(freq/1e10))[0][0])
        self.fIndsEval=np.sort(np.append(self.fIndsEval,self.solInds[-1]))
        self.fAxisGreedy_treeRefine(self.solInds[-1])
    
    def all_freqs(self):
        self.fIndsEval=np.array(range(len(self.fAxis)))

    def deactivate_timing(self):
        self.timing=False
        self.times={}

    def get_conv(self):
        resTot=np.linalg.norm(self.res_ROM[self.fIndsEval])/np.sqrt(len(self.fIndsEval))
        errTot=np.linalg.norm(self.err_R_F)/np.sqrt(len(self.fIndsTest))
        return resTot,errTot,self.res_ROM[self.fIndsEval],self.err_R_F

    def get_Zl(self, iBeam=None):
        if iBeam==None:
            Zl=[self.Zl[i][self.fIndsEval] for i in range(len(self.Zl))]
            return self.fAxis[self.fIndsEval],Zl
        else:
            return self.Zl[iBeam][self.fIndsEval]
    def get_Zt(self, iBeam):
        Ztest=self.get_Zl(iBeam)
        Zdriv=self.get_Zl(self.beams.getDrivingBeam())
        stest=np.array(self.beams.getPosition(iBeam))
        sdriv=np.array(self.beams.getPosition(self.beams.getDrivingBeam()))
        factor=getFelisConstants()[2]/self.fAxis[self.fIndsEval]/2/np.pi
        Zt=factor*(Ztest-Zdriv)/nlina.norm(stest-sdriv)
        return Zt

    # def getRHS(self,fInd):
    #     return self.RHS[:,fInd].toarray()[:,0]
    #
    # def getJSrc(self,fInd):
    #     return self.JSrc[:,fInd].toarray()[:,0]
    
    def get_fAxis(self):
        return self.fAxis

    def getDrivingBeam(self):
        return self.beams.getDrivingBeam()

    # def getJ(self,iBeam):
    #     return self.beams.getJ(iBeam)




    
    
    
    
    
    