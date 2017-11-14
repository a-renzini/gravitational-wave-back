import numpy as np
import qpoint as qp
import healpy as hp
import pylab
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

nside = 16
lmax = 2
test = False

#INTEGRATING FREQS:
low_f = 1.
high_f = 1024.  
low_cut = 80.
high_cut = 300.

#create empty lm object:
m_lm = np.ones(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)

#create object of class:
run = Ligo_Analyse(nside,lmax, fs, low_f, high_f)

# define start and stop time to search
# in GPS seconds
start = 931158100    #931079472: 31 segs   931158100: 69 segs  931168100: 7 segs 931184000:110 segs
stop  = 931622015 #971622015 #931622015 #931086336 #


####################################################################

print 'flagging the good data...'

segs_begin, segs_end = run.flagger(start,stop,filelist)

print len(segs_begin)

Z_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
S_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)

for sdx, (begin, end) in enumerate(zip(segs_begin,segs_end)):
    
    n=sdx+1
    
    print 'analysing segment number %s' % n

    print 'segmenting the data...'

    ctime, strain_H1, strain_L1 = run.segmenter(begin,end,filelist)
    stream = []
    
    
    if len(ctime)<2 : 
        print 'too short!'
        continue

    print 'segmenting done: ', len(ctime), ' segments'
    
    #SCANNING -> BUILDING FAKE TSTREAM

    '''
    
    for idx_t, ct_split in enumerate(ctime):
        stream.append(run.scan(ct_split, low_f, high_f, m_lm))

    '''

    #convenience chopping:

    #x = 30
    
    #ctime = ctime[:x]
    #strain_H1 = strain_H1[:x]
    #strain_L1 = strain_L1[:x]

    #print 'data conveniently chopped; analysing ', x, ' segments.'

    ####################################################################

    strain_split_H1 = []
    strain_split_L1 = []
    p_split_1 = []
    p_split_2 = []

    print 'filtering, ffting & saving the strains...'

    Nt = len(strain_H1[0])
    Nt = lf.bestFFTlength(Nt)
    
    for idx_str, (h1, l1) in enumerate(zip(strain_H1, strain_L1)):
        #print idx_str        
        # FT
        
        # plt.figure()
        # plt.plot(h1, color = 'r')
        # plt.plot(l1, color = 'b')
        # #plt.ylim([-100.,100.])
        # plt.savefig('testa.png' )
        
        #datah1 = 'datah1'
        #np.savez(datah1,h1)
        h1_in = h1.copy()
        l1_in = l1.copy()
        
        if test == True:
            h1_in = run.injector(h1_in,low_cut,high_cut)
            l1_in = run.injector(l1_in,low_cut,high_cut)
                
        strain_H1_f, H1_psd = run.filter(h1_in, low_cut,high_cut)
        strain_L1_f, L1_psd = run.filter(l1_in, low_cut,high_cut)
        
        freqs = np.fft.rfftfreq(2*Nt, 1./fs)
        freqs = freqs[:Nt/2+1]
        
        strain_H1_f *= 1.#/Nt
        strain_L1_f *= 1.#/Nt

        h1_psd = H1_psd(freqs)*fs**2   #(freqs)
        l1_psd = L1_psd(freqs)*fs**2
        
        #mask = (freqs>low_cut) & (freqs < high_cut)
        #print np.mean(h1_psd[mask]), np.min(h1_psd[mask]), np.max(h1_psd[mask])
        #print np.mean(l1_psd[mask]), np.min(l1_psd[mask]), np.max(l1_psd[mask])
        
        #WEIGHTING
        strain_H1_w = strain_H1_f/(h1_psd)
        strain_L1_w = strain_L1_f/(l1_psd)
        
        #print '=irfft='
        #hf1_inv = np.fft.irfft(strain_H1_f)
        #hf2_inv = np.fft.irfft(strain_L1_f)
        #print '++'
        
        '''
        now strain_x is a segment of 60 seconds of correlated signal, in frequency space.
        '''

        strain_split_H1.append(strain_H1_w)
        strain_split_L1.append(strain_L1_w)
    
        p_split_1.append(h1_psd) #strain_H1_coar * np.conj(strain_H1_coar)
        p_split_2.append(l1_psd)
    
    print 'filtering done'
    exit()
    #Integrate over frequency in the projector
        
    ####################################################################
        
    proj_lm = np.array([np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)]*len(ctime)) #why *len_ctime?

    print 'running the projector, obtaining a dirty map'
    
    print len(strain_split_H1), len(strain_split_H1[0])
      
    z_lm = run.projector(ctime,strain_split_H1, strain_split_L1,freqs)
    Z_lm +=z_lm
    
    for idx_t, ct_split in enumerate(ctime):
        ones = [1.]*len(freqs)
        proj_lm[idx_t] = run.summer(ctime[idx_t],ones,freqs) 
    
    #dirty_map_lm = hp.alm2map(np.sum(dt_lm,axis = 0),nside,lmax=lmax)
    dirty_map = hp.alm2map(z_lm,nside,lmax=lmax)
    hp.mollview(dirty_map)
    plt.savefig('maps_running/dirty_map%s.pdf' % sdx)
    #dirty_map_lm = hp.alm2map(np.sum(dt_lm,axis = 0),nside,lmax=lmax)

    print 'saved: dirty_map.pdf'

    ####################################################################
	
	
    #p_split_1 = H1_psd(freqs_x_coar) #strain_H1_coar * np.conj(strain_H1_coar)
    #p_split_2 = L1_psd(freqs_x_coar) #strain_L1_coar * np.conj(strain_L1_coar)
    
    print '+++'
    print np.mean(p_split_1[0])
    print np.mean(p_split_2[0])
    print '+++'
    
    # p_split_1_mid = []
    # p_split_2_mid = []
    #
    # for idx_p in range(len(p_split_1)):
    #     p_split_1_mid.append(np.mean(p_split_1[idx_p]))
    #     p_split_2_mid.append(np.mean(p_split_2[idx_p]))
    #
    # p_split_1_mid = np.array(p_split_1_mid)
    # p_split_2_mid = np.array(p_split_2_mid)

    print 'building M^-1:'

    M_lm_lpmp =[]
    
    print '1. scanning...'

    scan_lm = []
    
    
    for idx_t in range(len(ctime)):
            scan_lm.append(run.scanner(ctime[idx_t], p_split_1[idx_t],p_split_2[idx_t],freqs))

    print '2. projecting...'

    for idx_lm in range(hp.Alm.getidx(lmax,lmax,lmax)+1):
        
        M_lpmp = np.zeros(len(proj_lm[0]),dtype=complex)
        #print idx_lm
        if proj_lm[0][idx_lm]>EPSILON or proj_lm[0][idx_lm]<-EPSILON:
            for idx_t in range(len(ctime)):
                scan = scan_lm[idx_t]
                proj = proj_lm[idx_t][idx_lm]
                M_lpmp += np.conj(proj)*scan
        M_lm_lpmp.append(M_lpmp)

    print 'M is ', len(M_lm_lpmp), ' by ', len(M_lm_lpmp[0])
    
    np.savetxt('M%s.txt' % sdx,M_lm_lpmp)    
    
    print '3. inverting...'

    M_inv = np.linalg.inv(M_lm_lpmp)

    print 'the matrix has been inverted!'
    #print M_inv

    ####################################################################

    #s_lm = []
    s_p = []

#    for i in range(len(ctime)):
    s_lm = np.array(np.dot(M_inv,z_lm))
    S_lm+= s_lm
    s_p = hp.alm2map(s_lm,nside,lmax=lmax)
    #print len(s_lm)
    #print s_lm

    #dt_tot = np.sum(dt_lm,axis = 0)
    #print 'dt total:' , len(dt_tot.real)
    #print dt_tot

    hp.mollview(s_p)
    plt.savefig('maps_running/s_p%s.pdf' % sdx)
    
hp.mollview(hp.alm2map(Z_lm/len(ctime),nside,lmax=lmax))
plt.savefig('D_p%s.pdf' % sdx)

hp.mollview(hp.alm2map(S_lm/len(ctime),nside,lmax=lmax))
plt.savefig('S_p%s.pdf' % sdx)
    
exit()
    
    ############# using the decorrelator instead: #########
    #
    #
    # BIG_M = run.decorrelator(ctime, freqs_x_coar, p_split_1, p_split_2)
    # print 'BIG M is ', len(BIG_M), ' by ', len(BIG_M[0])
    # BIG_M_inv = np.linalg.inv(BIG_M)
    #
    # s_lm = np.array(np.dot(BIG_M_inv,z_lm))
    # s_p = hp.alm2map(s_lm,nside,lmax=lmax)
    #
    # hp.mollview(s_p)
    # plt.savefig('s_BIG_M_p.pdf')
    
    
    #plt.figure()
    #plt.axis([0,10000, 0.001, 10.]) 
    #plt.loglog()
    #plt.plot(fitted,label = 'fitted psd')
    #pylab.xlim([50.,500.])
    #pylab.ylim([0.,1E-43])
    #plt.xlabel('')
    #plt.ylabel('')
    #plt.legend()
    #plt.savefig('datapow.png')

