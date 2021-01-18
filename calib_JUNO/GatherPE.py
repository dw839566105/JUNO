import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

def LoadDataPE_TW(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    hess = 'hess' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def main_photon_sparse(path, order):
    ra = np.arange(1000, 18000, 1000)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%d' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return np.array(rd), coeff_pe

def main(order=5, fit_order=10):
    rd, coeff_pe = main_photon_sparse('JUNO_test/',order)
    coeff_L = np.zeros((order, fit_order + 1))
    
    for i in np.arange(order):
        if not i%2:
            B, tmp = np.polynomial.legendre.legfit(np.hstack((rd/np.max(rd),-rd/np.max(rd))), \
                                                      np.hstack((coeff_pe[i], coeff_pe[i])), \
                                                      deg = fit_order, full = True)
        else:
            B, tmp = np.polynomial.legendre.legfit(np.hstack((rd/np.max(rd),-rd/np.max(rd))), \
                                                      np.hstack((coeff_pe[i], -coeff_pe[i])), \
                                                      deg = fit_order, full = True)
            
        #B, tmp = np.polynomial.legendre.legfit(rd/np.max(rd), coeff_pe[i], deg = fit_order, full = True)
        y = np.polynomial.legendre.legval(rd/np.max(rd), B)

        coeff_L[i] = B

        plt.figure(num = i+1, dpi = 300)
        #plt.plot(rd, np.hstack((y2_in, y2_out[1:])), label='poly')
        plt.plot(np.array(rd)/1000, coeff_pe[i], 'r.', label='coefficient',linewidth=2)
        plt.plot(np.array(rd)/1000, y, label = 'fit')
        plt.xlabel('radius/m')
        plt.ylabel('PE Legendre coefficients')
        plt.title('%d-th, max fit = %d' %(i, fit_order))
        plt.legend()
        plt.savefig('coeff_PE_%d_%d_%d.png' % (i, order, fit_order))
        plt.close()
    return coeff_L

coeff_L = main(eval(sys.argv[1]), eval(sys.argv[2]))  
with h5py.File('./PE_coeff_1t_%d_%d.h5' % (eval(sys.argv[1]), eval(sys.argv[2])),'w') as out:
    out.create_dataset('coeff_L', data = coeff_L)
#A = np.polynomial.legendre.Legendre.fit(np.array((-0.5,0.4)), np.array((0.5,0.6)),deg=5)
