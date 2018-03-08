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
import MapBack as mb  #################
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

nside_in = 16
nside_out = 8
lmax = 2
sim = True  

#INTEGRATING FREQS:                                                                                                           
low_f = 80.
high_f = 300.
low_cut = 80.
high_cut = 300.

    
#DETECTORS
dects = ['H1','L1','V1']
ndet = len(dects)
nbase = int(ndet*(ndet-1)/2)
 
#create object of class:
run = mb.Telescope(nside_in,nside_out,lmax, fs, low_f, high_f, dects)


# define start and stop time to search
# in GPS seconds
start = 931035615 #S6 start GPS
stop  = 971622015 #971622015  #S6 end GPS



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
    S = 0
    N = 0
    print N
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
                print 'generating...'
                h1_in = my_h1.copy()
                l1_in = my_l1.copy()
                strains_in = (h1_in,l1_in)
                #print strains_in
                strains = run.injector(strains_in,my_ctime,low_cut,high_cut, sim)[0]
                #print len(strains)
                # plt.figure()
                # plt.plot((strains[0]))
                # plt.savefig('fakestreamsinv.pdf')
                # f = open('fakestreamsinv.txt', 'w')
                # for (i,x) in enumerate(strains[0]):
                #     print >>f, i, '     ', x
                # f.close()
                #pass the noisy strains to injector got the psds
            psds = run.injector(strains_copy,my_ctime,low_cut,high_cut)[1]
            #strains are the new generated strains
            aver = np.sqrt(np.abs(np.average(strains[0]*strains[1])))
            #print aver
            S += aver
            N += 1
            print 'N = ', N
            
            
          
            ctime_nproc = []
            strain1_nproc = []
            strain2_nproc = []
        
        #print idx_block    
        idx_block += 1
        #if idx_block == 1400: print S/N

    


  
    
    ##ssh -X ar6215@login.cx1.hpc.ic.ac.uk
