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
import MapBack as mb
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
start = 931079472    #931079472: 31 segs   931158100: 69 segs  931168100: 7 segs
stop  = 931622015 #971622015 #931622015 #931086336 #


####################################################################

print 'flagging the good data...'

segs_begin, segs_end = run.flagger(start,stop,filelist)

#print len(segs_begin)

if myid == 0: 
    my_segs_begin = split(segs_begin, nproc)
    my_segs_end = split(segs_end, nproc)
else:
    my_segs_begin = None
    my_segs_end = None


if ISMPI:
    my_segs_begin = comm.scatter(my_segs_begin)
    my_segs_end = comm.scatter(my_segs_end)

#create empty lm objects:
m_lm = np.ones(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)

my_M_lm_lpmp = 0.
    
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


for sdx, (begin, end) in enumerate(zip(my_segs_begin,my_segs_end)):
    
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
    import copy
    count = copy.deepcopy(len(ctime))

    ####################################################################

    strains_split = []
    psds_split = []

    print 'filtering, ffting & saving the strains...'

    Nt = len(strain_H1[0])
    Nt = lf.bestFFTlength(Nt)
    freqs = np.fft.rfftfreq(2*Nt, 1./fs)
    freqs = freqs[:Nt/2+1]
    mask = (freqs>low_f) & (freqs < high_f)
    #print run.sim_tstream(ctime[0],1.,1.,freqs)
    print 'filtering done'
    #Integrate over frequency in the projector
        
    ####################################################################
        
    #proj_lm = np.array([np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)]*len(ctime)) #why *len_ctime?

    
    # pix_bs = []
#     q_ns = []
#     pix_ns = []
#     m_lm = []
#
#     for i in range(len(ctime)):
#         pix_bs.append(run.geometry(ctime[i])[0])
#         q_ns.append(run.geometry(ctime[i])[1])
#         pix_ns.append(run.geometry(ctime[i])[2])
#         msave = run.m_lm(ctime[i], freqs[mask], pix_bs[i], q_ns[i])
#         m_lm.append(msave)
#         #print msave
#
#     Alm00 = np.zeros_like(m_lm[0])
#
#     for i in range(len(ctime)):
#         Alm00 += (m_lm[i]*np.conj(m_lm[i][0]))
#
#     print Alm00
#     Alm00_inv = []
#     for i in range(len(Alm00)):
#         Alm00_inv.append(1./Alm00[i]/len(Alm00))
#
#     print Alm00_inv
#
#     a00 = np.dot(Alm00_inv,Z_lm)
#     print a00


    M_lm_00 = M_lm_lpmp[0]        #symmetric => this should be fine
    Mlm00_inv = []
    for i in range(len(M_lm_00)):
        Mlm00_inv.append(1./M_lm_00[i] )
        print Mlm00_inv[i]*Z_lm[i]
         
    a00 = np.dot(Mlm00_inv,Z_lm)/len(M_lm_00)
    print a00
    M_lm_lpmp = np.real(M_lm_lpmp)
    M_inv = np.linalg.pinv(M_lm_lpmp)   #default:  for cond < 1E15
    S_lm = np.dot(M_inv,Z_lm)
    print S_lm
    print '==========='
    print '==========='
    
    frac = S_lm[0]
    i = 1
    while i <len(S_lm):
        frac += (M_lm_lpmp[0][i]*S_lm[i])/M_lm_lpmp[0][0]
        i+=1
    
    print frac
    exit()