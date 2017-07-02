import numpy as np
import qpoint as qp
import healpy as hp
#import ligo_analyse_class as lac
from ligo_analyse_class import Ligo_Analyse
import readligo as rl
import ligo_filter as lf
import matplotlib.pyplot as plt

EPSILON = 1E-24

####################################################################

# sampling rate:
fs = 4096
ligo_data_dir = '/Users/pai/Data/'  #can be defined in the repo
filelist = rl.FileList(directory=ligo_data_dir)

nside = 32
lmax = 8

run = Ligo_Analyse(nside,lmax)

# define start and stop time to search
# in GPS seconds
start = 931035615 #931079472 
stop  = 931086336 #931622015 #
#start = 931079472
#stop  = 931086336
#start = 931200000
#stop = 931300000

####################################################################

print 'segmenting the data...'

ctime, strain_H1, strain_L1 = run.segmenter(start,stop,filelist,fs)

print 'segmenting done: ', len(ctime), ' segments'

#convenience chopping:
#ctime = ctime[:2]
#strain_H1 = strain_H1[:2]
#strain_L1 = strain_L1[:2]

####################################################################

strain_H1_coar = []
strain_L1_coar = []
freq_x_coar = []

print 'coarse graining the strains...'

for idx_str, (h1, l1) in enumerate(zip(strain_H1, strain_L1)):

    # FT
    Nt = len(h1)
#    print Nt
    Nt = lf.bestFFTlength(Nt)
    strain_H1_f = np.fft.rfft(h1[:Nt])
    strain_L1_f = np.fft.rfft(l1[:Nt])

#    strain_x = strain_H1_f*np.conj(strain_L1_f)
#    power_x = strain_x*np.conj(strain_x)
    
    '''
    now strain_x is a segment of 60 seconds of correlated signal, in frequency space.
    '''

    # Coarse Grain

    s_H1_coar = []
    s_L1_coar = []
    f_x_coar = []
    
    sig_len = strain_H1_f.size
    print sig_len
    freq = np.fft.fftfreq(sig_len, d=1./fs)
    idx_a = 0

    while idx_a < sig_len:
        f_x_coar.append(np.mean(freq[idx_a:idx_a+10000])) #probably redundant
        s_H1_coar.append(np.mean(strain_H1_f[idx_a:idx_a+10000]))
        s_L1_coar.append(np.mean(strain_L1_f[idx_a:idx_a+10000]))
        idx_a = idx_a+10000

#    f_x_coar.append(np.mean(freq[idx_a-10000:]))
#    s_H1_coar.append(np.mean(strain_H1_f[idx_a-10000:]))
#    s_L1_coar.append(np.mean(strain_L1_f[idx_a-10000:]))
    
    f_x_coar = np.array(f_x_coar)
    s_H1_coar = np.array(s_H1_coar)
    s_L1_coar = np.array(s_L1_coar)
#    print s_x_coar
    
    strain_H1_coar.append(s_H1_coar)
    strain_L1_coar.append(s_L1_coar)
    freq_x_coar.append(f_x_coar)       
    
#s_x_coar = strain_H1_coar*np.conj(strain_L1_coar)
#p_x_coar = s_x_coar*np.conj(s_x_coar) #build this into a matrix (or put it into the projector routine?)


####################################################################

dt_lm = np.array([np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)]*len(ctime))
proj_lm = np.zeros_like(dt_lm)

print 'running the projector, obtaining a dirty map'

for idx_t in range(len(ctime)):
    dt_lm[idx_t] = run.projector(ctime, idx_t, strain_H1_coar, strain_L1_coar,freq_x_coar)
    ones = [1.]*len(freq_x_coar[idx_t])
    proj_lm[idx_t] = run.summer(ctime[idx_t],ones,freq_x_coar[idx_t])
  #N_t_tp*d_tp still not right

dirty_map_lm = hp.alm2map(np.sum(dt_lm,axis = 0),nside,lmax=lmax)
hp.mollview(dirty_map_lm)
plt.savefig('dirty_map_lm.pdf')

print 'saved: dirty_map_lm.pdf'


####################################################################

p_split_1 = strain_H1_coar * np.conj(strain_H1_coar)
p_split_2 = strain_L1_coar * np.conj(strain_L1_coar)

# p_split_1_mid = []
# p_split_2_mid = []
#
# for idx_p in range(len(p_split_1)):
#     p_split_1_mid.append(np.mean(p_split_1[idx_p]))
#     p_split_2_mid.append(np.mean(p_split_2[idx_p]))
#
# p_split_1_mid = np.array(p_split_1_mid)
# p_split_2_mid = np.array(p_split_2_mid)

M_lm_lpmp =[]

print 'building M^-1:'

print '1. scanning...'

scan_lm = []

for idx_t in range(len(ctime)):
        scan_lm.append(run.scanner_1(ctime,idx_t, p_split_1.real[idx_t],p_split_2.real,freq_x_coar))

print '2. projecting...'

for idx_lm in range(hp.Alm.getidx(lmax,lmax,lmax)+1):
        
    M_lpmp = np.zeros(len(dt_lm[0]),dtype=complex)
    #print idx_lm
    if proj_lm[0][idx_lm]>EPSILON or proj_lm[0][idx_lm]<-EPSILON:
        for idx_t in range(len(ctime)):
            scan = scan_lm[idx_t]
            proj = proj_lm[idx_t][idx_lm]
            #print proj
            M_lpmp += proj*scan
    M_lm_lpmp.append(M_lpmp)

print 'M is ', len(M_lm_lpmp), ' by ', len(M_lm_lpmp[0])

print '3. inverting...'

M_inv = np.linalg.inv(M_lm_lpmp)

print 'the matrix has been inverted!'
#print M_inv

####################################################################

#s_lm = []
s_p = []

for i in range(len(ctime)):
    s_lm = np.array(np.dot(M_inv,dt_lm[i]))
    s_p.append(hp.alm2map(s_lm,nside,lmax=lmax))
#print len(s_lm)
#print s_lm

#dt_tot = np.sum(dt_lm,axis = 0)
#print 'dt total:' , len(dt_tot.real)
#print dt_tot

hp.mollview(np.sum(s_p, axis = 0))
plt.savefig('s_p.pdf')