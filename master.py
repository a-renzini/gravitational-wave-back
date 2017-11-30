import numpy as np
import qpoint as qp
import healpy as hp
import pylab
#import ligo_analyse_class as lac
from ligo_analyse_class import Ligo_Analyse
import readligo as rl
import ligo_filter as lf
import matplotlib as mlb
import matplotlib.pyplot as plt
import time
import BigClass as bc

EPSILON = 1E-24


####################################################################
print '++++++++++++++++++++++++++'
print '=========================='
print '++++++++++++++++++++++++++'
print '=========================='
print (time.strftime("%H:%M:%S")), (time.strftime("%d/%m/%Y"))
print '++++++++++++++++++++++++++'
print '=========================='
print '++++++++++++++++++++++++++'
print '=========================='

# sampling rate:
fs = 4096
ligo_data_dir = '/Users/pai/Data/'  #can be defined in the repo
filelist = rl.FileList(directory=ligo_data_dir)

nside = 16
lmax = 2
sim = False

#INTEGRATING FREQS:
low_f = 80.
high_f = 300.
low_cut = 80.
high_cut = 300.

#DETECTORS
dects = ['H1','L1']#,'V1']
ndet = len (dects)
nbase = int(ndet*(ndet-1)/2)


#create empty lm objects:
m_lm = np.ones(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
M_lm_lpmp =0.

#create object of class:
run = bc.Telescope(nside,lmax, fs, low_f, high_f)

# define start and stop time to search
# in GPS seconds
start = 931079472    #931079472: 31 segs   931158100: 69 segs  931168100: 7 segs
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


    ####################################################################

    strains_split = []
    psds_split = []

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
        
        
        freqs = np.fft.rfftfreq(2*Nt, 1./fs)
        freqs = freqs[:Nt/2+1]
        
        # use a copy of the strains so that the filtering works smoothly
        strains = (h1,l1)
        strains_copy = (h1.copy(),l1.copy())
        
        ###########################
        
        
        if sim == True:
            h1_in = h1.copy()
            l1_in = l1.copy()
            strains = (h1_in,l1_in)
            strains = run.injector(strains,low_cut,high_cut, sim)[0]


        psds = run.injector(strains_copy,low_cut,high_cut)[1]
        
        strains_f = []
        psds_f = []
        strains_w = []
        
        for i in range(ndet):
            strains_f.append(run.filter(strains[i], low_cut,high_cut,psds[i]))
            psds_f.append(psds[i](freqs)*fs**2 )
            strains_w.append(strains_f[i]/(psds_f[i]))
        
        strains_split.append(strains_w)
        psds_split.append(psds_f)
        
        '''
        now strain_x is a segment of 60 seconds of correlated signal, in frequency space.
        '''

    
    #print '+++'
    #print run.sim_tstream(ctime[0],1.,1.,freqs)
    print 'filtering done'
    
    #Integrate over frequency in the projector
        
    ####################################################################
        
    #proj_lm = np.array([np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)]*len(ctime)) #why *len_ctime?

    print 'running the projector, obtaining a dirty map'
    
    pix_bs = []
    q_ns = []
    pix_ns = []
    
    for i in range(len(ctime)):
        pix_bs.append(run.geometry(ctime[i])[0])
        q_ns.append(run.geometry(ctime[i])[1])
        pix_ns.append(run.geometry(ctime[i])[2])
    

    
    z_lm = run.projector(ctime,strains_split,freqs,pix_bs, q_ns)
    Z_lm +=z_lm
    
   # for idx_t, ct_split in enumerate(ctime):
   #     ones = [1.]*len(freqs)
   #     proj_lm[idx_t] = run.summer(ctime[idx_t],ones,freqs) 
    
    #dirty_map_lm = hp.alm2map(np.sum(dt_lm,axis = 0),nside,lmax=lmax)
    dirty_map = hp.alm2map(Z_lm,nside,lmax=lmax)
    
    fig = plt.figure()
    hp.mollview(dirty_map)
    #hp.visufunc.projscatter(hp.pix2ang(nside,pix_bs))
    #hp.visufunc.projscatter(hp.pix2ang(nside,pix_ns))
    plt.savefig('maps_running/dirty_map%s.pdf' % sdx)
    #dirty_map_lm = hp.alm2map(np.sum(dt_lm,axis = 0),nside,lmax=lmax)

    print 'saved: dirty_map.pdf'
    
    
    ####################################################################
    
    print 'building M^-1:'
    
    for idx_t in range(len(ctime)):
        print idx_t
        M_lm_lpmp += run.M_lm_lpmp_t(ctime[idx_t], psds_split[idx_t],freqs,pix_bs[idx_t],q_ns[idx_t])
        #exit()

    print 'M is ', len(M_lm_lpmp), ' by ', len(M_lm_lpmp[0])
    
    
    f = open('Ms_running/M%s.txt' % sdx, 'w')
    print >>f, M_lm_lpmp
    print >>f, '===='
    print >>f, np.linalg.eigh(M_lm_lpmp)
    print >>f, '===='
    print >>f, np.linalg.cond(M_lm_lpmp)
    f.close
    
    print '3. inverting...'

    M_inv = np.linalg.inv(M_lm_lpmp)

    print 'the matrix has been inverted!'
    
    #exit()

    #print M_inv

    ####################################################################

    #s_lm = []
    S_p = []

    S_lm = np.array(np.dot(M_inv,Z_lm)) #fully accumulated maps!
    #S_lm+= s_lm
    S_p = hp.alm2map(S_lm,nside,lmax=lmax)
    #print len(s_lm)
    #print s_lm
    
    print np.mean(S_p)
    
    #dt_tot = np.sum(dt_lm,axis = 0)
    #print 'dt total:' , len(dt_tot.real)
    #print dt_tot

    hp.mollview(S_p)
    plt.savefig('maps_running/S_p%s.pdf' % sdx)
        
hp.mollview(hp.alm2map(Z_lm/len(ctime),nside,lmax=lmax))
plt.savefig('Z_p%s.pdf' % sdx)

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


'''
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

print M_lm_lpmp

print '3. inverting...'

M_inv = np.linalg.inv(M_lm_lpmp)

print 'the matrix has been inverted!'
'''
    
    
    ##ssh -X ar6215@login.cx1.hpc.ic.ac.uk
