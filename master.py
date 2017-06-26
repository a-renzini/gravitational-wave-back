import numpy as np
import qpoint as qp
import healpy as hp
#import ligo_analyse_class as lac
from ligo_analyse_class_2 import Ligo_Analyse
import readligo as rl
import matplotlib.pyplot as plt


# sampling rate:
fs = 4096
ligo_data_dir = '/Users/pai/Data/'  #can be defined in the repo
filelist = rl.FileList(directory=ligo_data_dir)

nside = 32
lmax = 16

run = Ligo_Analyse(nside,lmax)

# define start and stop time to search
# in GPS seconds
start = 931035615 #931079472 
stop  = 931622015 #931086336
#start = 931079472
#stop  = 931086336
#start = 931200000
#stop = 931300000


ctime, strain_H1, strain_L1 = run.segmenter(start,stop,filelist,fs)

print 'segmenting done'

#strain_H1_f = np.zeros_like(strain_H1)
#strain_L1_f = np.zeros_like(strain_L1)

strain_x_coar = []
power_x_coar = []
freq_x_coar = []

for idx_str, (h1, l1) in enumerate(zip(strain_H1, strain_L1)):

    # FT
    Nt = len(h1)
#    print Nt
    strain_H1_f = np.fft.rfft(h1[:Nt])
    strain_L1_f = np.fft.rfft(l1[:Nt])

    strain_x = strain_H1_f*np.conj(strain_L1_f)
    power_x = strain_x*np.conj(strain_x)
    
    '''
    now strain_x is a segment of 60 seconds of correlated signal, in frequency space.
    '''

    # Coarse Grain

    s_x_coar = []
    p_x_coar = []
    f_x_coar = []
    
    sig_len = strain_x.size
    print sig_len
    freq = np.fft.fftfreq(sig_len, d=1./fs)
    idx_a = 0
    idx_coar = 0
    while idx_a < sig_len:
        f_x_coar.append(np.mean(freq[idx_a:idx_a+10000]))
        s_x_coar.append(np.mean(strain_x[idx_a:idx_a+10000]))
        p_x_coar.append(np.mean(power_x[idx_a:idx_a+10000]))
        idx_a = idx_a+10000
        idx_coar+=1
        if idx_a+10000 > sig_len:
            f_x_coar.append(np.mean(freq[idx_a:]))
            s_x_coar.append(np.mean(strain_x[idx_a:]))
            p_x_coar.append(np.mean(power_x[idx_a:]))

    strain_x_coar.append(s_x_coar)
    power_x_coar.append(p_x_coar)
    freq_x_coar.append(f_x_coar)       
        
print len(freq_x_coar)
print len(freq_x_coar[1])
print len(strain_x_coar)
print len(strain_x_coar[1])

out = []

for i in range(len(ctime)):
    out.append(run.projector(ctime[i], strain_x_coar[i], power_x_coar[i],freq_x_coar[i]))

#print len[out]
print len(out[13].real)

hp.mollview(out[13].real)
plt.savefig('map13.pdf')

hp.mollview(np.sum(out, axis=0).real)
plt.savefig('map.pdf')

    
