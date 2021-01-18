#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tables
from scipy.spatial.distance import pdist, squareform
import os


# In[86]:


rr = []
xx = []
yy = []
zz = []
EE = []
for radius in np.arange(-17000, 18000, 1000):
    x = np.zeros(0)
    y = np.zeros(0)
    z = np.zeros(0)
    E = np.zeros(0)
    for file in np.arange(0,20):
        try:
            h = tables.open_file('./result_old/%d/%02d.h5' % (radius,file),'r')
            recondata = h.root.Recon
            E1 = recondata[:]['E_sph_in']
            x1 = recondata[:]['x_sph_in']
            y1 = recondata[:]['y_sph_in']
            z1 = recondata[:]['z_sph_in']
            L1 = recondata[:]['Likelihood_in']
            s1 = recondata[:]['success_in']

            E2 = recondata[:]['E_sph_out']
            x2 = recondata[:]['x_sph_out']
            y2 = recondata[:]['y_sph_out']
            z2 = recondata[:]['z_sph_out']
            L2 = recondata[:]['Likelihood_out']
            s2 = recondata[:]['success_out']

            data = np.zeros((np.size(x1),4))

            index = L1 < L2
            data[index,0] = x1[index]
            data[index,1] = y1[index]
            data[index,2] = z1[index]
            data[index,3] = E1[index]

            data[~index,0] = x2[~index]
            data[~index,1] = y2[~index]
            data[~index,2] = z2[~index]
            data[~index,3] = E2[~index]

            x = np.hstack((x, data[(s1 * s2)!=0,0]))
            y = np.hstack((y, data[(s1 * s2)!=0,1]))
            z = np.hstack((z, data[(s1 * s2)!=0,2]))
            E = np.hstack((E, data[(s1 * s2)!=0,3]))
            h.close()
        except:
            pass
    try:
        if(x.shape[0]>0):
            xx.append(np.array((np.mean(x), np.std(x))))
            yy.append(np.array((np.mean(y), np.std(y))))
            zz.append(np.array((np.mean(z), np.std(z))))
            EE.append(np.array((np.mean(E/2000), np.std(E/2000))))
            rr.append(radius)
    except:
        pass


# In[87]:


plt.plot(rr, np.array(zz)/1.029,'.')


# In[88]:


print(np.array(zz)[:,0])
print(np.array(rr))
print(np.array(zz)[:,0]*1000*17/17.5/np.array(rr))


# In[89]:


(zz[-1][0]/17.5*17)


# In[15]:


np.array()


# In[64]:


h = tables.open_file('../calib_JUNO/base.h5')
print(h.root.base[:])
h.close()


# In[63]:


h.root.base[:]


# In[78]:


E_mean = []
E_std = []
for i in np.arange(-17000,18000,1000):
    data = np.loadtxt('test%d.txt' % i)
    E_mean.append(np.mean(data))
    E_std.append(np.std(data))


# In[79]:


plt.plot(np.arange(-17000,18000,1000), E_std)


# In[84]:


np.loadtxt('total.txt').shape


# In[85]:


np.loadtxt('total_Ein.txt').shape


# In[107]:


rr = []
xx = []
yy = []
zz = []
EE = []
for radius in np.arange(-17000, 18000, 1000):
    x = np.zeros(0)
    y = np.zeros(0)
    z = np.zeros(0)
    E = np.zeros(0)
    for file in np.arange(0,20):
        try:
            h = tables.open_file('./result/%d/%02d.h5' % (radius,file),'r')
            recondata = h.root.Recon
            E1 = recondata[:]['E_sph_in']
            x1 = recondata[:]['x_sph_in']
            y1 = recondata[:]['y_sph_in']
            z1 = recondata[:]['z_sph_in']
            L1 = recondata[:]['Likelihood_in']
            s1 = recondata[:]['success_in']

            E2 = recondata[:]['E_sph_out']
            x2 = recondata[:]['x_sph_out']
            y2 = recondata[:]['y_sph_out']
            z2 = recondata[:]['z_sph_out']
            L2 = recondata[:]['Likelihood_out']
            s2 = recondata[:]['success_out']

            data = np.zeros((np.size(x1),4))

            index = L1 < L2
            data[index,0] = x1[index]
            data[index,1] = y1[index]
            data[index,2] = z1[index]
            data[index,3] = E1[index]

            data[~index,0] = x2[~index]
            data[~index,1] = y2[~index]
            data[~index,2] = z2[~index]
            data[~index,3] = E2[~index]

            x = np.hstack((x, data[(s1 * s2)!=0,0]))
            y = np.hstack((y, data[(s1 * s2)!=0,1]))
            z = np.hstack((z, data[(s1 * s2)!=0,2]))
            E = np.hstack((E, data[(s1 * s2)!=0,3]))
            h.close()
        except:
            pass
    try:
        if(x.shape[0]>0):
            xx.append(np.array((np.mean(x), np.std(x))))
            yy.append(np.array((np.mean(y), np.std(y))))
            zz.append(np.array((np.mean(z), np.std(z))))
            EE.append(np.array((np.mean(E), np.std(E))))
            rr.append(radius)
    except:
        pass


# In[109]:


plt.figure(dpi=300)
plt.plot(rr, np.array(zz)[:,0] - np.array(rr)/1000,'.')


# In[101]:


plt.plot(rr, np.array(EE)[:,1],'.')


# In[183]:


plt.figure(dpi=300, num=1)
plt.figure(dpi=300, num=2)
plt.figure(dpi=300, num=3)
plt.figure(dpi=300, num=4)

import numpy as np, scipy.stats as st

# returns confidence interval of mean
def confIntMean(a, conf=0.95):
    mean, sem, m = np.mean(a), st.sem(a), st.t.ppf((1+conf)/2., len(a)-1)
    return m*sem
    #return mean - m*sem, mean + m*sem

import scipy.stats
rr = []
xx = []
yy = []
zz = []
EE = []
for radius in np.arange(-18000, 18000, 1000):
    x = np.zeros(0)
    y = np.zeros(0)
    z = np.zeros(0)
    E = np.zeros(0)
    for file in np.arange(0,20):
        try:
            h = tables.open_file('./result/%d/%02d.h5' % (radius,file),'r')
            #print('./result/%d/%02d.h5'% (radius,file))
            recondata = h.root.Recon
            E1 = recondata[:]['E_sph_in']
            x1 = recondata[:]['x_sph_in']
            y1 = recondata[:]['y_sph_in']
            z1 = recondata[:]['z_sph_in']
            L1 = recondata[:]['Likelihood_in']
            s1 = recondata[:]['success_in']

            E2 = recondata[:]['E_sph_out']
            x2 = recondata[:]['x_sph_out']
            y2 = recondata[:]['y_sph_out']
            z2 = recondata[:]['z_sph_out']
            L2 = recondata[:]['Likelihood_out']
            s2 = recondata[:]['success_out']

            data = np.zeros((np.size(x1),4))

            index = L1 < L2
            data[index,0] = x1[index]
            data[index,1] = y1[index]
            data[index,2] = z1[index]
            data[index,3] = E1[index]

            data[~index,0] = x2[~index]
            data[~index,1] = y2[~index]
            data[~index,2] = z2[~index]
            data[~index,3] = E2[~index]

            x = np.hstack((x, data[(s1 * s2)!=0,0]))
            y = np.hstack((y, data[(s1 * s2)!=0,1]))
            z = np.hstack((z, data[(s1 * s2)!=0,2]))
            E = np.hstack((E, data[(s1 * s2)!=0,3]))
            h.close()
        except:
            pass
    r = np.sqrt(x**2 + y**2 + z**2)*np.sign(z) * 1000
    plt.figure(num = 1)
    plt.errorbar(radius/1000, np.mean(r) - radius, 
                 #yerr=np.std(r)/np.sqrt(r.shape[0]-1),
                 yerr = confIntMean(r),
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
    plt.figure(num = 2)
    tmp = st.norm.interval(0.95, loc=np.mean(r), scale=st.sem(r))
    plt.errorbar(radius/1000, np.std(r), 
                 yerr=(tmp[1] - tmp[0])/2,
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
    plt.figure(num = 3)
    plt.errorbar(radius/1000, np.mean(E),
                 yerr = confIntMean(E),
                 #yerr=np.std(E)/np.sqrt(r.shape[0]-1), 
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
    plt.figure(num = 4)
    tmp = st.norm.interval(0.95, loc=np.mean(E), scale=st.sem(E))
    plt.errorbar(radius/1000, np.std(E),
                 yerr=(tmp[1] - tmp[0])/2,
                 #yerr=np.std(E)/np.sqrt(r.shape[0]-1), 
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
#print(scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1)) 
plt.figure(num = 1)
plt.xlabel('Radius/m')
plt.ylabel(r'Mean bias of $R_{rec} - R_{truth}$ [mm]')
plt.savefig('total_bias.png')
plt.figure(num = 2)
plt.xlabel('Radius/m')
plt.ylabel(r'Resolution of $R_{rec} - R_{truth}$ [mm]')
plt.savefig('total_res.png')
plt.figure(num = 3)
plt.xlabel('Radius/m')
plt.ylabel(r'Mean bias of $E_{rec}$ [MeV]')
plt.savefig('total_bias_E.png')
plt.figure(num = 4)
plt.xlabel('Radius/m')
plt.ylabel(r'Resolution of $E_{rec}$ [MeV]')
plt.savefig('total_res_E.png')


# In[185]:


plt.figure(dpi=300, num=1)
plt.figure(dpi=300, num=2)
plt.figure(dpi=300, num=3)
plt.figure(dpi=300, num=4)

import numpy as np, scipy.stats as st

# returns confidence interval of mean
def confIntMean(a, conf=0.95):
    mean, sem, m = np.mean(a), st.sem(a), st.t.ppf((1+conf)/2., len(a)-1)
    return m*sem
    #return mean - m*sem, mean + m*sem

import scipy.stats
rr = []
xx = []
yy = []
zz = []
EE = []
for radius in np.arange(1000, 18000, 1000):
    x = np.zeros(0)
    y = np.zeros(0)
    z = np.zeros(0)
    E = np.zeros(0)
    for file in np.arange(0,20):
        try:
            h = tables.open_file('./result/%d/%02d.h5' % (radius,file),'r')
            #print('./result/%d/%02d.h5'% (radius,file))
            recondata = h.root.Recon
            E1 = recondata[:]['E_sph_in']
            x1 = recondata[:]['x_sph_in']
            y1 = recondata[:]['y_sph_in']
            z1 = recondata[:]['z_sph_in']
            L1 = recondata[:]['Likelihood_in']
            s1 = recondata[:]['success_in']

            E2 = recondata[:]['E_sph_out']
            x2 = recondata[:]['x_sph_out']
            y2 = recondata[:]['y_sph_out']
            z2 = recondata[:]['z_sph_out']
            L2 = recondata[:]['Likelihood_out']
            s2 = recondata[:]['success_out']

            data = np.zeros((np.size(x1),4))

            index = L1 < L2
            data[index,0] = x1[index]
            data[index,1] = y1[index]
            data[index,2] = z1[index]
            data[index,3] = E1[index]

            data[~index,0] = x2[~index]
            data[~index,1] = y2[~index]
            data[~index,2] = z2[~index]
            data[~index,3] = E2[~index]

            x = np.hstack((x, data[(s1 * s2)!=0,0]))
            y = np.hstack((y, data[(s1 * s2)!=0,1]))
            z = np.hstack((z, data[(s1 * s2)!=0,2]))
            E = np.hstack((E, data[(s1 * s2)!=0,3] * 2))
            h.close()
        except:
            pass
    r = np.sqrt(x**2 + y**2 + z**2)*np.sign(z) * 1000
    plt.figure(num = 1)
    plt.errorbar(radius/1000, np.mean(r) - radius, 
                 #yerr=np.std(r)/np.sqrt(r.shape[0]-1),
                 yerr = confIntMean(r),
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
    plt.figure(num = 2)
    tmp = st.norm.interval(0.95, loc=np.mean(r), scale=st.sem(r))
    plt.errorbar(radius/1000, np.std(r), 
                 yerr=(tmp[1] - tmp[0])/2,
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
    plt.figure(num = 3)
    plt.errorbar(radius/1000, np.mean(E),
                 yerr = confIntMean(E),
                 #yerr=np.std(E)/np.sqrt(r.shape[0]-1), 
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
    plt.figure(num = 4)
    tmp = st.norm.interval(0.95, loc=np.mean(E), scale=st.sem(E))
    plt.errorbar(radius/1000, np.std(E),
                 yerr=(tmp[1] - tmp[0])/2,
                 #yerr=np.std(E)/np.sqrt(r.shape[0]-1), 
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
#print(scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1)) 
plt.figure(num = 1)
plt.xlabel('Radius/m')
plt.ylabel(r'Mean bias of $R_{rec} - R_{truth}$ [mm]')
plt.savefig('total_bias_pos.png')
plt.figure(num = 2)
plt.xlabel('Radius/m')
plt.ylabel(r'Resolution of $R_{rec} - R_{truth}$ [mm]')
plt.savefig('total_res_pos.png')
plt.figure(num = 3)
plt.xlabel('Radius/m')
plt.ylabel(r'Mean bias of $E_{rec}$ [MeV]')
plt.savefig('total_bias_E_pos.png')
plt.figure(num = 4)
plt.xlabel('Radius/m')
plt.ylabel(r'Resolution of $E_{rec}$ [MeV]')
plt.savefig('total_res_E_pos.png')


# In[228]:


plt.figure(dpi=300, num=1)
plt.figure(dpi=300, num=2)
plt.figure(dpi=300, num=3)
plt.figure(dpi=300, num=4)
import scipy.stats
rr = []
xx = []
yy = []
zz = []
EE = []
for radius in np.arange(1000, 18000, 1000):
    x = np.zeros(0)
    y = np.zeros(0)
    z = np.zeros(0)
    E = np.zeros(0)
    for file in np.arange(0,20):
        try:
            h = tables.open_file('./result/%d/%02d.h5' % (radius,file),'r')
            #print('./result/%d/%02d.h5'% (radius,file))
            recondata = h.root.Recon
            E1 = recondata[:]['E_sph_in']
            x1 = recondata[:]['x_sph_in']
            y1 = recondata[:]['y_sph_in']
            z1 = recondata[:]['z_sph_in']
            L1 = recondata[:]['Likelihood_in']
            s1 = recondata[:]['success_in']

            E2 = recondata[:]['E_sph_out']
            x2 = recondata[:]['x_sph_out']
            y2 = recondata[:]['y_sph_out']
            z2 = recondata[:]['z_sph_out']
            L2 = recondata[:]['Likelihood_out']
            s2 = recondata[:]['success_out']

            data = np.zeros((np.size(x1),4))

            index = L1 < L2
            data[index,0] = x1[index]
            data[index,1] = y1[index]
            data[index,2] = z1[index]
            data[index,3] = E1[index]

            data[~index,0] = x2[~index]
            data[~index,1] = y2[~index]
            data[~index,2] = z2[~index]
            data[~index,3] = E2[~index]

            x = np.hstack((x, data[(s1 * s2)!=0,0]))
            y = np.hstack((y, data[(s1 * s2)!=0,1]))
            z = np.hstack((z, data[(s1 * s2)!=0,2]))
            E = np.hstack((E, data[(s1 * s2)!=0,3]))
            h.close()
        except:
            pass
    r = np.sqrt(x**2 + y**2 + z**2)*np.sign(z) * 1000
    plt.figure(num = 1)
    plt.errorbar(radius/1000, np.mean(r) - radius, 
                 yerr=np.std(r)/np.sqrt(r.shape[0]-1), 
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
    plt.figure(num = 2)
    plt.errorbar(radius/1000, np.std(r), 
                 #yerr=np.std(E)/np.sqrt(r.shape[0]-1), 
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
    plt.figure(num = 3)
    plt.errorbar(radius/1000, np.mean(E),
                 yerr = confIntMean(E),
                 #yerr=np.std(E)/np.sqrt(r.shape[0]-1), 
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
    plt.figure(num = 4)
    tmp = st.norm.interval(0.95, loc=np.mean(E), scale=st.sem(E))
    plt.errorbar(radius/1000, np.std(E),
                 yerr=(tmp[1] - tmp[0])/2,
                 #yerr=np.std(E)/np.sqrt(r.shape[0]-1), 
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
#print(scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1)) 
plt.figure(num = 1)
plt.xlabel('Radius/m')
plt.ylabel(r'Mean bias of $R_{rec} - R_{truth}$ [mm]')
plt.savefig('total_bias_positive.png')
plt.figure(num = 2)
plt.xlabel('Radius/m')
plt.ylabel(r'Resolution of $R_{rec} - R_{truth}$ [mm]')
plt.savefig('total_res_positive.png')


# In[186]:


0.03/np.sqrt(2)


# In[199]:


h = tables.open_file('result/-10000/00.h5')
recondata = h.root.Recon
E1 = recondata[:]['E_sph_in']
x1 = recondata[:]['x_sph_in']
y1 = recondata[:]['y_sph_in']
z1 = recondata[:]['z_sph_in']
L1 = recondata[:]['Likelihood_in']
s1 = recondata[:]['success_in']

E2 = recondata[:]['E_sph_out']
x2 = recondata[:]['x_sph_out']
y2 = recondata[:]['y_sph_out']
z2 = recondata[:]['z_sph_out']
L2 = recondata[:]['Likelihood_out']
s2 = recondata[:]['success_out']

data = np.zeros((np.size(x1),4))

index = L1 < L2
data[index,0] = x1[index]
data[index,1] = y1[index]
data[index,2] = z1[index]
data[index,3] = E1[index]

data[~index,0] = x2[~index]
data[~index,1] = y2[~index]
data[~index,2] = z2[~index]
data[~index,3] = E2[~index]
h.close()

print(np.std(data, axis=0))
print(np.mean(data, axis=0))


# In[200]:


np.mean(z1)


# In[203]:


np.std(np.loadtxt('test.log'))


# In[207]:


plt.hist(np.loadtxt('test.log')*2, bins=500)
plt.show()


# In[232]:


plt.figure(dpi=300, num=1)
plt.figure(dpi=300, num=2)
import scipy.stats
rr = []
xx = []
yy = []
zz = []
EE = []
for radius in np.arange(4000, 8000, 1000):
    x = np.zeros(0)
    y = np.zeros(0)
    z = np.zeros(0)
    E = np.zeros(0)
    for file in np.arange(0,2):
        h = tables.open_file('./result/%d/%02d.h5' % (radius,file),'r')
        #print('./result/%d/%02d.h5'% (radius,file))
        recondata = h.root.Recon
        E1 = recondata[:]['E_sph_in']
        x1 = recondata[:]['x_sph_in']
        y1 = recondata[:]['y_sph_in']
        z1 = recondata[:]['z_sph_in']
        L1 = recondata[:]['Likelihood_in']
        s1 = recondata[:]['success_in']

        data = np.zeros((np.size(x1),4))
        
        r1 = np.sqrt(x1**2 + y1**2 + z1**2)
        index = np.abs(r1)<200
        x = np.hstack((x, x1[index]))
        y = np.hstack((y, y1[index]))
        z = np.hstack((z, z1[index]))
        E = np.hstack((E, E1[index]))
        h.close()

    r = np.sqrt(x**2 + y**2 + z**2)*np.sign(z) * 1000
    plt.figure(num = 1)
    plt.errorbar(radius/1000, np.mean(r) - radius, 
                 yerr=np.std(r)/np.sqrt(r.shape[0]-1), 
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
    plt.figure(num = 2)
    plt.errorbar(radius/1000, np.std(r), 
                 #yerr=np.std(E)/np.sqrt(r.shape[0]-1), 
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
    plt.figure(num = 3)
    plt.errorbar(radius/1000, np.mean(E),
                 yerr = confIntMean(E),
                 #yerr=np.std(E)/np.sqrt(r.shape[0]-1), 
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
    plt.figure(num = 4)
    tmp = st.norm.interval(0.95, loc=np.mean(E), scale=st.sem(E))
    plt.errorbar(radius/1000, np.std(E),
                 yerr=(tmp[1] - tmp[0])/2,
                 #yerr=np.std(E)/np.sqrt(r.shape[0]-1), 
                 #yerr = scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1),
                 marker='s',
                 mfc='red',ms=3,ecolor='green')
#print(scipy.stats.sem(r)*scipy.stats.t.ppf((1 + 0.95) / 2, r.shape[0]-1)) 
plt.figure(num = 1)
plt.xlabel('Radius/m')
plt.ylabel(r'Mean bias of $R_{rec} - R_{truth}$ [mm]')
plt.savefig('total_bias_positive.png')
plt.figure(num = 2)
plt.xlabel('Radius/m')
plt.ylabel(r'Resolution of $R_{rec} - R_{truth}$ [mm]')
plt.savefig('total_res_positive.png')


# In[231]:


for radius in np.arange(2000, 3000, 1000):
    x = np.zeros(0)
    y = np.zeros(0)
    z = np.zeros(0)
    E = np.zeros(0)
    for file in np.arange(0,2):
        h = tables.open_file('./result/%d/%02d.h5' % (radius,file),'r')
        #print('./result/%d/%02d.h5'% (radius,file))
        recondata = h.root.Recon
        E1 = recondata[:]['E_sph_in']
        x1 = recondata[:]['x_sph_in']
        y1 = recondata[:]['y_sph_in']
        z1 = recondata[:]['z_sph_in']
        L1 = recondata[:]['Likelihood_in']
        s1 = recondata[:]['success_in']

        data = np.zeros((np.size(x1),4))
        
        index = np.abs(z1)<10
        x = np.hstack((x, x1[index]))
        y = np.hstack((y, y1[index]))
        z = np.hstack((z, z1[index]))
        E = np.hstack((E, E1[index]))
        h.close()
        
plt.hist(z,bins=100)
plt.show()


# In[ ]:




