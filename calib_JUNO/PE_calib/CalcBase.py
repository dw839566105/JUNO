import numpy as np
import h5py
import tables

data_path = '/junofs/users/junoprotondecay/xubd/harvest/det/e-/0_0_1/2/'


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
    
    PMT_pos = np.vstack((A[:,0],x,y,z,GGain))
    return PMT_pos.T, Gtype

def readfile(filename):
    h1 = tables.open_file(filename,'r')
    print(filename)
    truthtable = h1.root.photoelectron
    EventID = truthtable[:]['TriggerNo']
    ChannelID = truthtable[:]['ChannelID']
    h1.close()
    
    size = PMT_pos[:,0].shape[0]
    PE = np.zeros(size*(np.unique(EventID).shape[0]))
    for j in np.unique(EventID):
        j = np.int(j)
        Q = np.bincount(ChannelID[EventID==j])
        x = np.zeros(np.int(np.max(PMT_pos[:,0]))+1)
        x[0:Q.shape[0]] = Q
        PE[j*size:(j+1)*size] = x[PMT_pos[:,0].astype('int')]
    
    return EventID, ChannelID, PE

def readchain(data_path):
    for i in np.arange(0, 20):
        if(i == 0):
            filename = data_path+'0/%02d.h5' % i
            EventID, ChannelID, PE = readfile(filename)
        else:
            try:
                filename = data_path+'0/%02d.h5' % i
                EventID1, ChannelID1, PE1 = readfile(filename)
                EventID = np.hstack((EventID, EventID1))
                ChannelID = np.hstack((ChannelID, ChannelID1))
                PE = np.hstack((PE, PE1))
            except:
                pass

    return EventID, ChannelID, PE

def CalMean():
    data_path = '/junofs/users/junoprotondecay/xubd/harvest/det/e-/0_0_1/2/'
    data = []
    
    EventID, ChannelID, PE = readchain(data_path)
    S = np.mean(np.reshape(PE, (-1, PMT_pos[:,0].shape[0])),axis=0)
    Gain = np.zeros_like(PMT_pos[:,0])
    print(PMT_pos)
    for name in np.unique(Gtype):
        Gain[Gtype==name] = np.mean(S[Gtype==name])/np.mean(PMT_pos[Gtype==name, -1])*PMT_pos[Gtype==name, -1]
        #Gain[Gtype==name] = np.mean(S[Gtype==name])
    print(PE.shape, S.shape, np.reshape(PE, (-1, PMT_pos[:,0].shape[0])).shape)
    return Gain
PMT_pos, Gtype = ReadPMT()
Gain= CalMean()
print(Gain)
#mean = (size_x*ax0 + size_y*ay0 + size_z*az0)/(size_x+size_y+size_z)
#base = np.exp(np.mean(np.log(mean)))
#correct = mean/base
#print(correct)

with h5py.File('base1.h5','w') as out:
    out.create_dataset('base', data = Gain)
#    out.create_dataset('correct', data = correct)