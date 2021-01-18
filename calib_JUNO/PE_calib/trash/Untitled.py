#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tables
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


data_path = '/junofs/users/junoprotondecay/xubd/harvest/det/e-/0_0_1/2/'

h = tables.open_file(data_path+'-15000/00.h5')
TriggerNo = h.root.photoelectron[:]['TriggerNo']
ChannelID = h.root.photoelectron[:]['ChannelID']
HitPosInWindow = h.root.photoelectron[:]['HitPosInWindow']
h.close()


# In[ ]:





# In[267]:


A = np.loadtxt('/junofs/users/junoprotondecay/xubd/harvest/data/geo.csv')


# In[6]:


x = 17.5 * np.sin(A[:,1]) * np.sin(A[:,2])
y = 17.5 * np.sin(A[:,1]) * np.cos(A[:,2])
z = 17.5 * np.cos(A[:,1])
np.vstack((x,y,z)).shape


# In[270]:


plt.hist(A[:,1], bins=100)
plt.show()
plt.hist(A[:,2], bins=100)
plt.show()


# In[7]:


np.max(ChannelID)


# In[8]:


A.shape


# In[9]:


np.unique(ChannelID).shape


# In[10]:


np.max(A)


# In[15]:


small = A[:,0]<100000
A[small,0].shape
big = A[:,0]>100000
A[big,0].shape


# In[17]:


A[big]


# In[46]:


for i in np.unique(TriggerNo):
    Q = np.bincount(ChannelID[TriggerNo==i].astype('int'))
    x = np.zeros(np.max(A[:,0]).astype('int')+1)
    x[0:Q.shape[0]] = Q
    print(np.sum(x[(A[:,0]).astype('int')]), np.sum(TriggerNo==i))


# In[23]:


(A[:,0]).astype('int')


# In[107]:





# In[157]:


G = np.loadtxt('/cvmfs/juno.ihep.ac.cn/sl6_amd64_gcc830/Pre-Release/J20v1r0-Pre2/data/Simulation/ElecSim/pmtdata.txt',dtype=bytes).astype('str')
G = np.setdiff1d(G[:,0].astype('int'),A[:,0])
for i in np.unique(G[:,1]):
    plt.figure(dpi=300)
    plt.hist(S[G[:,1] == i])
    plt.show()


# In[122]:


print(A.shape)
print(G.shape)


# In[153]:


np.argwhere(G[:,0].astype('int') != A[:,0])
np.setdiff1d(G[:,0].astype('int'),A[:,0])


# In[204]:


data_path = '/junofs/users/junoprotondecay/xubd/harvest/det/e-/0_0_1/2/'
A = np.loadtxt('/junofs/users/junoprotondecay/xubd/harvest/data/geo.csv')
x = 17.5 * np.sin(A[:,1]) * np.sin(A[:,2])
y = 17.5 * np.sin(A[:,1]) * np.cos(A[:,2])
z = 17.5 * np.cos(A[:,1])

TriggerNo = np.zeros(0)
ChannelID = np.zeros(0)
#HitPosInWindow = np.zeros(0)

PE_total = np.zeros(0)
size = A[:,0].shape[0]
A[:,0] = A[:,0].astype('int')
for i in np.arange(0,20):
    print(i)
    h = tables.open_file(data_path+'0/%02d.h5' % i)
    #TriggerNo = np.hstack((TriggerNo, h.root.photoelectron[:]['TriggerNo'].astype('int')))
    TriggerNo = h.root.photoelectron[:]['TriggerNo'].astype('int')
    ChannelID = h.root.photoelectron[:]['ChannelID'].astype('int')
    #ChannelID = np.hstack((ChannelID, h.root.photoelectron[:]['ChannelID'].astype('int')))
    #HitPosInWindow = np.hstack((HitPosInWindow, h.root.photoelectron[:]['HitPosInWindow']))

    PE = np.zeros(size*(np.unique(TriggerNo).shape[0]))
    for j in np.unique(TriggerNo):
        j = np.int(j)
        Q = np.bincount(ChannelID[TriggerNo==j])
        x = np.zeros(np.int(np.max(A[:,0]))+1)
        x[0:Q.shape[0]] = Q
        PE[j*size:(j+1)*size] = x[A[:,0].astype('int')]
    h.close()
    PE_total = np.hstack((PE_total, PE))


# In[130]:


S = np.mean(np.reshape(PE_total, (40000,-1)),axis=0)


# In[145]:





# In[138]:


plt.plot(G[:,2].astype(float))


# In[61]:


h1 = tables.open_file(data_path+'0/12.h5')
h1.root.photoelectron[:]['TriggerNo']


# In[147]:


print(np.sum([ChannelID>300000]))
print(np.sum([ChannelID<300000]))


# In[140]:


S.shape


# In[148]:


np.sum(S==0)


# In[155]:


A[17610:17618]


# In[156]:


G[17610:17618]


# In[182]:


G[:,0] = G[:,0].astype('int')
from numpy.lib import recfunctions as rfn
print(rfn.__doc__)


# In[175]:


A1 = A
G1 = G


# In[217]:


GG = G[:,0].astype('int')

id1 = np.setdiff1d(GG,A[:,0])
#G1 = G[GG=id1]
#G1[:,0] = G[G[:,0].astype('int')!=id1,0].astype('int')
Gtype = G[GG!=id1,1]
GGain = G[GG!=id1,2].astype('float')


# In[219]:


Gain = np.zeros_like(GGain)
for name in np.unique(Gtype):
    plt.figure(dpi=300)
    plt.hist(S[Gtype==name], label='%.3f' % np.mean(S[Gtype==name]))
    Gain[Gtype==name] = np.mean(S[Gtype==name])/np.mean(GGain[Gtype==name])*GGain[Gtype==name]
    plt.title(name)
    plt.legend()
    plt.show()
    
plt.plot(Gain)


# In[220]:





# In[225]:


plt.hist(S/Gain, bins=30)
plt.show()


# In[295]:


data = []
for i in np.arange(-17000,18000, 1000):
    h = tables.open_file('../JUNO1/file_%d.h5' % i)
    _ = h.root.coeff25[:]
    #if(i==0):
    #    a = np.sum(_)
    #    _ = np.zeros_like(_)
    #    _[0] = a
    data.append(_)
    h.close()


# In[296]:


for i in np.arange(30):
    plt.figure(dpi=300)
    plt.plot(np.arange(-17,18,1), np.array(data)[:,i],'.')
    plt.title(f'{i}-th order')
    plt.show()


# In[300]:


aic = []
for i in np.arange(-17000,18000, 1000):
    h = tables.open_file('../JUNO/file_%d.h5' % i)
    data = []
    for j in np.arange(5, 30):
        data.append(eval('h.root.AIC%d[()]' % j))
    aic.append(np.where(np.array(data) == np.min(data))[0][0] + 5)
    h.close()


# In[301]:


plt.plot(aic,'.')


# In[ ]:




