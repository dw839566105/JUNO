'''
This is the heart of total PE calib
The PE calib program use Poisson regression, which is equal to decomposition of log expect
since the problem is spherical symmetric, we use Legendre polynomial

    1. each PMT position 'x' and fixed vertex(or vertexes with the same radius r) 'v'
    2. transform 'v' to (0,0,z) as an axis
    3. do the same transform to x, we got the zenith angle 'theta' and azimuth angle 'phi', 'phi' can be ignored
    4. calculate the different Legendre order of 'theta', as a big matrix 'X' (PMT No. * order)
    5. expected 'y' by total pe, got GLM(generalize linear model) 'X beta = g(y)', 'beta' is a vector if coefficient, 'g' is link function, we use log
    6. each r has a set of 'beta' for later analysis
    
The final optimize using scipy.optimize instead of sklear.**regression
'''

import numpy as np 
import scipy, h5py
import tables
import sys
import time
from scipy.optimize import minimize
from scipy.optimize import rosen_der
#from numpy.polynomial import legendre as LG
import matplotlib.pyplot as plt
from scipy.linalg import norm
#from numdifftools import Jacobian, Hessian
from sklearn.linear_model import Lasso
from sklearn.linear_model import TweedieRegressor
import statsmodels.api as sm
from numba import jit
@jit(nopython=True)
def legval(x, c):
    """
    stole from the numerical part of numpy.polynomial.legendre

    """
    if len(c) == 1:
        return c[0]
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1*(nd - 1))/nd
            c1 = tmp + (c1*x*(2*nd - 1))/nd
    return c0 + c1*x

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def LoadBase():
    '''
    # to vanish the PMT difference, just a easy script
    # output: relative different bias
    '''
    path = './base1.h5'
    h1 = tables.open_file(path)
    base = h1.root.base[:]
    h1.close()
    return base
# after using the same PMT this part can be omitted
# base = np.log(LoadBase()) # dont forget log

def ReadPMT():
    A = np.loadtxt('/junofs/users/junoprotondecay/xubd/harvest/data/geo.csv')
    x = 17.5 * np.sin(A[:,1]/180*np.pi) * np.cos(A[:,2]/180*np.pi)
    y = 17.5 * np.sin(A[:,1]/180*np.pi) * np.sin(A[:,2]/180*np.pi)
    z = 17.5 * np.cos(A[:,1]/180*np.pi)
    
    Gdata = np.loadtxt('/cvmfs/juno.ihep.ac.cn/sl6_amd64_gcc830/Pre-Release/J20v1r0-Pre2/data/Simulation/ElecSim/pmtdata.txt',dtype=bytes).astype('str')
    G = np.setdiff1d(Gdata[:,0].astype('int'),A[:,0])
    
    GG = Gdata[:,0].astype('int')
    id1 = np.setdiff1d(GG,A[:,0])
    
    Gtype = Gdata[GG!=id1,1]
    GGain = Gdata[GG!=id1,2].astype('float')
    Gain = np.zeros_like(GGain)
    #for name in np.unique(Gtype):
    #    Gain[Gtype==name] = np.mean(S[Gtype==name])/np.mean(GGain[Gtype==name])*GGain[Gtype==name]
        
    PMT_pos = np.vstack((A[:,0],x,y,z,Gain))
    return PMT_pos.T, Gtype

def Calib(theta, *args):
    '''
    # core of this program
    # input: theta: parameter to optimize
    #      *args: include 
          total_pe: [event No * PMT size] * 1 vector ( used to be a 2-d matrix)
          PMT_pos: PMT No * 3
          cut: cut off of Legendre polynomial
          LegendreCoeff: Legendre value of the transformed PMT position (Note, it is repeated to match the total_pe)
    # output: L : likelihood value
    '''
    total_pe, PMT_pos, cut, LegendreCoeff = args
    y = total_pe
    #corr = np.dot(LegendreCoeff, theta) + np.tile(base, (1, np.int(np.size(LegendreCoeff)/np.size(base)/np.size(theta))))[0,:]
    corr = np.dot(LegendreCoeff, theta)
    # Poisson regression as a log likelihood
    # https://en.wikipedia.org/wiki/Poisson_regression
    L0 = - np.sum(np.sum(np.transpose(y)*np.transpose(corr) \
        - np.transpose(np.exp(corr))))
    # how to add the penalty? see
    # http://jmlr.csail.mit.edu/papers/volume17/15-021/15-021.pdf
    # the following 2 number is just a good attempt
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
    rho = 1
    alpha = 0
    L = L0/(2*np.size(y)) + alpha * rho * norm(theta,1) + 1/2* alpha * (1-rho) * norm(theta,2) # elastic net
    return L0

def Legendre_coeff(PMT_pos_, vertex, cut):
    '''
    # calulate the Legendre value of transformed X
    # input: PMT_pos: PMT No * 3
          vertex: 'v' 
          cut: cut off of Legendre polynomial
    # output: x: as 'X' at the beginnig    
    
    '''
    size = np.size(PMT_pos_[:,0])
    cos_theta = np.sum(vertex*PMT_pos_,axis=1)\
            /np.sqrt(np.sum(vertex**2, axis=1)*np.sum(PMT_pos_**2,axis=1))
    cos_theta[np.isnan(cos_theta)] = 1 # for v in detector center
    '''
    x = np.zeros((size, cut))
    # legendre coeff
    for i in np.arange(0,cut):
        c = np.zeros(cut)
        c[i] = 1
        x[:,i] = legval(cos_theta,c)
    '''
    x = legval(cos_theta, np.eye(cut).reshape((cut,cut,1))).T
    print(PMT_pos_.shape, x.shape, cos_theta.shape)
    return x, cos_theta

def readfile(filename):
    h1 = tables.open_file(filename,'r')
    print(filename)
    truthtable = h1.root.photoelectron
    EventID = truthtable[:]['TriggerNo']
    ChannelID = truthtable[:]['ChannelID']
    h1.close()
    
    size = PMT_pos[:,0].shape[0]
    PE = np.zeros(size)
    #PE = np.zeros(size*(np.unique(EventID).shape[0]))
    for j in np.unique(EventID):
        j = np.int(j)
        Q = np.bincount(ChannelID[EventID==j])
        x = np.zeros(np.int(np.max(PMT_pos[:,0]))+1)
        x[0:Q.shape[0]] = Q

        PE += x[PMT_pos[:,0].astype('int')]
    
    return EventID, ChannelID, PE

def readchain(data_path):
    for i in np.arange(0, 2):
        if(i == 0):
            filename = data_path+'%02d.h5' % i
            EventID, ChannelID, PE = readfile(filename)
        else:
            try:
                filename = data_path+'%02d.h5' % i
                EventID1, ChannelID1, PE1 = readfile(filename)
                EventID = np.hstack((EventID, EventID1))
                ChannelID = np.hstack((ChannelID, ChannelID1))
                PE = np.hstack((PE, PE1))
            except:
                pass

    return EventID, ChannelID, PE

def main_Calib(radius,fout, cut_max, PMT_pos):
    '''
    # main program
    # input: radius: %+.3f, 'str' (in makefile, str is default)
    #        path: file storage path, 'str'
    #        fout: file output name as .h5, 'str' (.h5 not included')
    #        cut_max: cut off of Legendre
    # output: the gathered result EventID, ChannelID, x, y, z
    '''
    print('begin reading file', flush=True)
    #filename = '/mnt/stage/douwei/Simulation/1t_root/1.5MeV_015/1t_' + radius + '.h5'
    with h5py.File(fout,'w') as out:
        # read files by table
        # we want to use 6 point on different axis to calib
        # +x, -x, +y, -y, +z, -z or has a little lean (z axis is the worst! use (0,2,10) instead) 
        # In ceter, no positive or negative, the read program should be changed
        # In simulation, the (0,0,0) saved as '*-0.000.h5' is negative
        # positive direction
        data_path = '/junofs/users/junoprotondecay/xubd/harvest/det/e-/0_0_1/2/'
        PMTNo = np.size(PMT_pos[:,0])
        #x = np.array((0,0,eval(radius)))
        x = np.array((0,0,eval(radius)))
        EventID, ChannelID, PE = readchain(data_path + radius + '/')
        EventID1, ChannelID1, PE1 = readchain(data_path + '-' + radius + '/')
        EventID = np.hstack((EventID, EventID1))
        ChannelID = np.hstack((ChannelID, ChannelID1))
        PE = np.hstack((PE, PE1))


        EventNo = np.int(PE.shape[0]/PMT_pos[:,0].shape[0])
        print(EventNo)
        #S = np.mean(np.reshape(PE, (-1, PMT_pos[:,0].shape[0])),axis=0)
        #PE = np.reshape(PE, (-1, PMT_pos[:,0].shape[0]))
        #total_pe = S
        
        print('begin processing legendre coeff', flush=True)
        # this part for the same vertex
        tmp = time.time()

        vertex = np.tile(np.atleast_2d(x), (PMTNo, 1))
        Legend_, cos_theta = Legendre_coeff(PMT_pos[:,1:4], vertex, cut_max)
        LegendreCoeff = np.tile(Legend_, (np.int(EventNo/2), 1))
        cos = np.tile(cos_theta, (1, np.int(EventNo/2))).T[:,0]
        
        vertex1 = np.tile(-np.atleast_2d(x), (PMTNo, 1))
        Legend_, cos_theta = Legendre_coeff(PMT_pos[:,1:4], vertex1, cut_max)
        LegendreCoeff1 = np.tile(Legend_, (np.int(EventNo/2), 1))
        cos1 = np.tile(cos_theta, (1, np.int(EventNo/2))).T[:,0]

        vertex = np.hstack((vertex, vertex1))
        LegendreCoeff = np.vstack((LegendreCoeff, LegendreCoeff1))
        print(LegendreCoeff.shape, LegendreCoeff1.shape)
        cos = np.hstack((cos, cos1))

        offset = np.tile(base, (1, EventNo)).T[:,0]

        for cut in np.arange(5,cut_max,1): # just take special values
            X = LegendreCoeff[:,0:cut]
            y = PE
            #y = PE.T
            print(X.shape, y.shape, offset.shape)
            model = sm.GLM(y, X, offset=offset, family=sm.families.Poisson())
            result = model.fit()
            print(result.predict(X))
            #plt.figure(dpi=300)
            #plt.plot(cos, result.predict(X),'.')
            #plt.plot(cos_theta, total_pe,'.')
            #plt.plot(cos_theta, S/np.exp(base),'.') 
            #plt.hist(cos[:80000], weights=(PE/np.exp(offset))[:80000],bins=50)
            #plt.savefig('tmp'+ radius + '.png')
            print(result.summary())
            L = result.aic
            coeff = result.params
            std = result.bse
            print(result.aic)
            print(result.bic)
            out.create_dataset('coeff' + str(cut), data = coeff)
            out.create_dataset('std' + str(cut), data = std)
            out.create_dataset('AIC' + str(cut), data = L)

if len(sys.argv)!=4:
    print("Wront arguments!")
    print("Usage: python main_calib.py 'radius' outputFileName[.h5] Max_order")
    sys.exit(1)
    
PMT_pos, Gtype = ReadPMT()
PMT_pos = PMT_pos[:,0:4]
base = np.log(LoadBase())
# sys.argv[1]: '%s' radius
# sys.argv[2]: '%s' path
# sys.argv[3]: '%s' output
# sys.argv[4]: '%d' cut
main_Calib(sys.argv[1], sys.argv[2], eval(sys.argv[3]), PMT_pos)
