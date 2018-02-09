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
plt.switch_backend('agg')
import time
import MapBackM0 as mb  #################
from mpi4py import MPI
ISMPI = True
#if mpi4py not present: ISMPI = False

import os
import sys

data_path = sys.argv[1]
out_path =  sys.argv[2]

try:
    sys.argv[3]
except (NameError, IndexError):
    checkpoint = None
else:
    checkpoint = True
    checkfile_path = sys.argv[3]

    
if os.path.exists(data_path):
    print 'the data its in the ' , data_path
    # file exists                                                                                                             
if os.path.exists(out_path):
    print 'output goes to ' , out_path
    
###############                                                                                                               

def split(container, count):
    """                                                                                                                       
    Simple function splitting a container into equal length chunks.                                                           
    Order is not preserved but this is potentially an advantage depending on                                                  
    the use case.                                                                                                             
    """
    return [container[_i::count] for _i in range(count)]

###############                                                                                                               

EPSILON = 1E-24

if ISMPI:

    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    myid = comm.Get_rank()


else:
    comm = None
    nproc = 1
    myid = 0

print 'myid: {} of {}'.format(myid,nproc)

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
ligo_data_dir = data_path  #can be defined in the repo                                                                        
filelist = rl.FileList(directory=ligo_data_dir)


nside = 8
lmax = 2
sim = True  

#INTEGRATING FREQS:                                                                                                           
low_f = 80.
high_f = 300.
low_cut = 80.
high_cut = 300.

    
#DETECTORS
dects = ['H1','L1']#,'V1']
ndet = len (dects)
nbase = int(ndet*(ndet-1)/2)
 
#create object of class:
run = mb.Telescope(nside,lmax, fs, low_f, high_f)

# define start and stop time to search
# in GPS seconds
start = 931035615 #S6 start GPS
stop  = 971622015  #S6 end GPS


###########################UNCOMMENT ME#########################################

print 'flagging the good data...'

segs_begin, segs_end = run.flagger(start,stop,filelist)

ctime_nproc = []
strain1_nproc = []
strain2_nproc = []
b_pixes = []

if myid == 0:
    Z_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
    S_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
    M_lm_lpmp = 0.
    counter = 0
    conds = []
    if checkpoint  == True:
        checkdata = np.load(checkfile_path)
        Z_lm += checkdata['Z_lm']
        M_lm_lpmp += checkdata['M_lm_lpmp']
        S_lm = None
        counter = checkdata['counter']
        conds = checkdata['conds']

else:
    Z_lm = None
    S_lm = None
    M_lm_lpmp = None
    counter = 0

print 'segmenting the data...'

mlm = []

for sdx, (begin, end) in enumerate(zip(segs_begin,segs_end)):

    n=sdx+1

    ctime, strain_H1, strain_L1 = run.segmenter(begin,end,filelist)

    if len(ctime)<2 : continue
    
    idx_block = 0
    while idx_block < len(ctime):
        ctime_nproc.append(ctime[idx_block])
        strain1_nproc.append(strain_H1[idx_block])
        strain2_nproc.append(strain_L1[idx_block])
        
        if len(ctime_nproc) == nproc:   #################################
                        #
            ######################################################
            ######################################################
            ######################################################

            idx_list = np.arange(nproc)

            if myid == 0:
                my_idx = np.split(idx_list, nproc)  #probably redundant .. could just say my_ctime = ctime[myid]

            else:
                my_idx = None


            if ISMPI:
                my_idx = comm.scatter(my_idx)
                my_ctime = ctime_nproc[my_idx[0]]
                my_h1 = strain1_nproc[my_idx[0]]
                my_l1 = strain2_nproc[my_idx[0]]


            #create empty lm objects:
            m_lm = np.ones(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
            my_M_lm_lpmp = 0.

            print 'filtering, ffting & saving the strains...'

            Nt = len(my_h1)
            Nt = lf.bestFFTlength(Nt)

            freqs = np.fft.rfftfreq(2*Nt, 1./fs)
            freqs = freqs[:Nt/2+1]
            
            mask = (freqs>low_f) & (freqs < high_f)
            
            strains = (my_h1,my_l1)
            strains_copy = (my_h1.copy(),my_l1.copy()) #calcualte psds from these

                    ###########################


            if sim == True:
                h1_in = my_h1.copy()
                l1_in = my_l1.copy()
                strains_in = (h1_in,l1_in)
                strains = run.injector(strains_in,low_cut,high_cut, sim)[0]
                
                #pass the noisy strains to injector got the psds
            psds = run.injector(strains_copy,low_cut,high_cut)[1]

            strains_f = []
            psds_f = []
            strains_w = []

            for i in range(ndet):
                strains_f.append(run.filter(strains[i], low_cut,high_cut,psds[i]))
                psds_f.append(psds[i](freqs)*fs**2 )
                strains_w.append(strains_f[i]/(psds_f[i]))


            '''
            now strains_w, etc are pairs of 60s segments of signal, in frequency space.
            '''


                #print '+++'
                #print run.sim_tstream(ctime[0],1.,1.,freqs)
            print 'filtering done'
                #Integrate over frequency in the projector

            ####################################################################

            #proj_lm = np.array([np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)]*len(ctime)) #why *len_ctime?

            print 'running the projector, obtaining a dirty map'

            pix_bs = run.geometry(my_ctime)[0]
            q_ns = run.geometry(my_ctime)[1]
            pix_ns = run.geometry(my_ctime)[2]
            
            print pix_bs
                        
            for i in range(len(pix_bs)):
                b_pixes.append(pix_bs[i])
            
            
            m_lm = run.mbuilder(my_ctime,strains_w,freqs,pix_bs, q_ns)
            z_lm = run.projector(my_ctime,strains_w,freqs,pix_bs, q_ns)
            