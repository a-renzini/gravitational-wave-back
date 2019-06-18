import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.special import spherical_jn, sph_harm    
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz, hanning
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py
import datetime as dt
import pytz
import pylab
import qpoint as qp
import healpy as hp
from camb.bispectrum import threej
import quat_rotation as qr
from scipy.optimize import curve_fit
import OverlapFunctsSrc as ofs
from numpy import cos,sin
from matplotlib import cm
from mpi4py import MPI
ISMPI = True

# LIGO-specific readligo.py 
import readligo as rl
import ligo_filter as lf
from gwpy.time import tconvert
from glue.segments import *

import MapBack_2 as mb
import time
import math
#if mpi4py not present: ISMPI = False

import os
import sys
import notches as notchfile

def PDX(frexx,a,b,c):
    #b = 1.
    #c = 1.
     
    return (a*1.e-22*((18.*b)/(0.1+frexx))**2)**2+(0.07*a*1.e-22)**2+((frexx/(2000.*c))*.4*a*1.e-22)**2


def notches(run_name):
    
    if run_name == 'O1':#34.70, 35.30,  #LIVNGSTON: 33.7 34.7 35.3 
        #notch_fs = np.array([ 34.70, 35.30,35.90, 36.70, 37.30, 40.95, 60.00, 120.00, 179.99, 304.99, 331.9, 499.0, 500.0, 510.02,  1009.99])
        notch_fs = notchfile.no_O1 
        
    if run_name == 'O2':             
        #notch_fs = np.array([30.25, 31.25,32.25,33.0,34.5,35.25,36.25,37.0,40.5,41.75,45.5,46.0,59.6,299.5,305.0,315.4,331.5,500.25])
        notch_fs = notchfile.no_O2
        
    return notch_fs
    
def sigmas(run_name):

    if run_name == 'O1':
        sigma_fs = notchfile.sig_O1
        
    if run_name == 'O2':                         
        sigma_fs = notchfile.sig_O2            
    
    return sigma_fs    
  
def Pdx_notcher(freqx,Pdx,run_name):
    mask = np.ones_like(freqx, dtype = bool)

    for (idx_f,f) in enumerate(freqx):
        for i in range(len(notches(run_name))):
            if f > (notches(run_name)[i]-15.*sigmas(run_name)[i]) and f < (notches(run_name)[i]+15.*sigmas(run_name)[i]):
                mask[idx_f] = 0
                
    return freqx[mask],Pdx[mask]

def Pdx_nanner(freqx,Pdx):
    mask = np.ones_like(freqx, dtype = bool)

    for (idx_f,f) in enumerate(freqx):
        for i in range(len(notches())):
            if f > (notches()[i]-2.5*sigmas()[i]) and f < (notches()[i]+2.5*sigmas()[i]):
                mask[idx_f] = 0.
                
    return freqx*mask,Pdx*mask

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def halfgaussian(x, mu, sig):
    out = np.ones_like(x)
    out[int(mu):]= np.exp(-np.power(x[int(mu):] - mu, 2.) / (2 * np.power(sig, 2.)))
    return out                    

def PSD_params(strain1, low_f, high_f,run_name):
    
    Nt = len(strain1)
    Nt = lf.bestFFTlength(Nt)
    
    fs=4096       
    dt=1./fs
    
    freqs = np.fft.rfftfreq(Nt, 1./fs)

    # frequency mask

    mask = (freqs>low_f) & (freqs < high_f)
    
    strain_in_1 = strain1

    '''WINDOWING & RFFTING.'''

    Nt = len(strain_in_1)
    Nt = lf.bestFFTlength(Nt)


    strain_in = strain_in_1[:Nt]
    strain_in_cp = np.copy(strain_in)
    
    strain_in_nowin = np.copy(strain_in)
    strain_in_nowin *= signal.tukey(Nt,alpha=0.05)
    strain_in_cp *= signal.tukey(Nt,alpha=0.05)

    freqs = np.fft.rfftfreq(Nt, dt)
    hf_nowin = np.fft.rfft(strain_in_nowin, n=Nt, norm = 'ortho') #####!HERE! 03/03/18 #####

    fstar = fs

    Psd_data = abs(hf_nowin.copy()*np.conj(hf_nowin.copy())) 

    mask = (freqs>low_f) & (freqs < high_f)
    

    if high_f < 300. or low_f<30.:
        masxx = (freqs>30.) & (freqs < 300.)
                        
    else:        
        masxx = (freqs>low_f) & (freqs < high_f)
    
    freqs_cp = np.copy(freqs)
    Psd_cp = np.copy(Psd_data)
    freqs_cp = freqs_cp[masxx]
    Psd_cp = Psd_cp[masxx]
    
    freqs_notch,Psd_notch = Pdx_notcher(freqs_cp,Psd_cp,run_name)
    freqcp = np.copy(freqs_notch)
    Pscp = np.copy(Psd_notch)


    try:
        fit = curve_fit(PDX, freqcp, Pscp) #, bounds = ([0.,0.,0.],[2.,2.,2.])) 
        psd_params = fit[0]

    except RuntimeError:
        print myid, "Error - curve_fit failed"
        psd_params = [10.,10.,10.]
           
    # plt.figure()
    #
    # plt.loglog(freqs[mask], np.abs(hf_nowin[mask])**2, label = 'nowin PSD')
    # plt.loglog(freqs[mask], hf_psd(freqs[mask])*1., label = 'mlab PSD')
    #
    # plt.loglog(frexcp, Pxcp, label = 'notchy PSD')
    #
    # plt.loglog(frexx[masxx],PDX(frexx,a,b,c)[masxx], label = 'notched pdx fit')
    # plt.legend()
    # plt.show()

    # plt.figure()
    #
    # plt.loglog(freqs[mask], np.abs(hf_nowin[mask])**2, label = 'nowin PSD')
    # plt.loglog(freqs[mask], hf_psd(freqs[mask])*1., label = 'mlab PSD')
    #
    # plt.loglog(frexcp, Pxcp, label = 'notchy PSD')
    #
    # plt.loglog(frexx[masxx],PDX(frexx,a,b,c)[masxx], label = 'notched pdx fit')
    # plt.legend()
    # plt.savefig('seg.png')

    
    return psd_params

#FROM THE SHELL: data path, output path, type of input map, SNR level (noise =0, high, med, low)

data_path = sys.argv[1]
out_path =  sys.argv[2]
maptyp = '1pole'
noise_lvl = 1
noise_lvl = int(noise_lvl)
this_path = out_path

FULL_DESC = True
cnt = 0

# poisson masked "flickering" map
poi = False

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


# MPI setup for run 

if ISMPI:

    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    myid = comm.Get_rank()


else:
    comm = None
    nproc = 1
    myid = 0

if myid == 0:

    params = []
    paramsl = []
    endtimes = []
        
    minute = 0

else:
    
    params = None
    paramsl = None

    minute = None


# sampling rate; resolutions in/out
                                                                                                              
fs = 4096
nside_in = 32
nside_out = 8
npix_out = hp.nside2npix(nside_out)

# load the LIGO file list

ligo_data_dir = data_path                                                                         
filelist = rl.FileList(directory=ligo_data_dir)


# declare whether to simulate (correlated) data (in frequency space)
sim = False

# frequency cuts (integrate over this range)
                                                                                                          
low_f = 30.
high_f = 500.

# spectral shape of the GWB

alpha = 3. 
f0 = 100.

# DETECTORS (should make this external input)

dects = ['H1','L1']
ndet = len(dects)
nbase = int(ndet*(ndet-1)/2)
avoided = 0 

ctime_nproc = []
strain1_nproc = []
strain2_nproc = []


# GAUSSIAN SIM. INPUT MAP CASE: make sure that the background map isn't re-simulated between scans, 
# and between checkfiles 


# INITIALISE THE CLASS  ######################
# args of class: nsides in/out; sampling frequency; freq cuts; declared detectors; the path of the checkfile; SNR level


run = mb.Telescope(nside_in,nside_out, fs, low_f, high_f, dects, maptyp,this_path,noise_lvl,alpha,f0)

##############################################


##########################  RUN  TIMES  #########################################

# RUN TIMES : define start and stop time to search in GPS seconds; 
# if checkpoint = True make sure to start from end of checkpoint

counter = 0         #counter = number of mins analysed
bads = 0

run_name = 'O1'

if run_name == 'O1':
    start = 1126224017#1164956817  #start = start time of O1 : 1126224017    1450000000  #1134035217 probs
    stop  = 1137254417#1187733618  #1127224017       #1137254417  #O1 end GPS     

elif run_name == 'O2':
    start = 1164956817  #start = start time of O1 : 1126224017    1450000000  #1134035217 probs
    stop  = 1187733618  #1127224017       #1137254417  #O1 end GPS     

else: print 'run?'

##########################################################################

########################### data  massage  #################################

if myid == 0:
    print 'flagging'
    segs_begin, segs_end = run.flagger(start,stop,filelist)
    segs_begin = list(segs_begin)
    segs_end = list(segs_end)


    i = 0
    while i in np.arange(len(segs_begin)):
        delta = segs_end[i]-segs_begin[i]
        if delta > 15000:   #250 min
            steps = int(math.floor(delta/15000.))
            for j in np.arange(steps):
                step = segs_begin[i]+(j+1)*15000
                segs_end[i+j:i+j]=[step]
                segs_begin[i+j+1:i+j+1]=[step]
            i+=steps+1
        else: i+=1

else: 
    segs_begin = None
    segs_end = None

segs_begin = comm.bcast(segs_begin, root=0)
segs_end = comm.bcast(segs_end, root=0)



for sdx, (begin, end) in enumerate(zip(segs_begin,segs_end)):
    
    n=sdx+1
    
    # ID = 0 segments the data
    
    if myid == 0:
        
        ctime, strain_H1, strain_L1 = run.segmenter(begin,end,filelist)
        len_ctime = len(ctime)
        
    else: 
        ctime = None
        strain_H1 = None
        strain_L1 = None
        len_ctime = None
        len_ctime_nproc = None    
    
    len_ctime = comm.bcast(len_ctime, root=0)
    
    if len_ctime<2 : continue      #discard short segments (may up this to ~10 mins)
    
    
    #idx_block: keep track of how many mins we're handing out
    
    idx_block = 0

    while idx_block < len_ctime:
        
        # accumulate ctime, strain arrays of length exactly nproc 
        
        if myid == 0:
            ctime_nproc.append(ctime[idx_block])
            strain1_nproc.append(strain_H1[idx_block])
            strain2_nproc.append(strain_L1[idx_block])
            
            len_ctime_nproc = len(ctime_nproc)
        # iminutes % nprocs == rank
        
        len_ctime_nproc = comm.bcast(len_ctime_nproc, root=0)
        
        if len_ctime_nproc == nproc:
   
            idx_list = np.arange(nproc)

            if myid == 0:
                my_idx = np.split(idx_list, nproc)  
                
            else:
                my_idx = None


            if ISMPI:
                my_idx = comm.scatter(my_idx)
                my_ctime = comm.scatter(ctime_nproc)#ctime_nproc[my_idx[0]]
                my_h1 = comm.scatter(strain1_nproc)
                my_l1 = comm.scatter(strain2_nproc)
                my_endtime = my_ctime[-1]
            
            
            ctime_idx = my_ctime
            strain1 = my_h1
            strain2 = my_l1

            strains = (strain1,strain2)
            strains_copy = (strain1.copy(),strain2.copy()) #calcualte psds from these
            
            params1 = PSD_params(strains_copy[0],low_f,high_f,run_name)
            params2 = PSD_params(strains_copy[1],low_f,high_f,run_name)
                        
            if myid == 0:
                
                #PSD1_setbuf = nproc * [np.zeros_like(fr_psd_1)]
                #PSD2_setbuf = nproc * [np.zeros_like(fr_psd_1)]
                endtimes_buff = nproc *[0]
                endtime = 0                

                norms_buff = nproc *[0]
                params_buff = nproc * [np.zeros_like(params1)]
                
                minute += nproc
                
            else: 
                
                #PSD1_setbuf = None
                #PSD2_setbuf = None
                endtimes_buff = None
                endtime = None   
                
                avoided_buff = None
                
                if FULL_DESC == True:
                    
                    norms_buff = None
                    params_buff = None 
                    normsl_buff = None
                    paramsl_buff = None 
            
            if ISMPI: 
                                
                comm.barrier()
                
                #PSD1_setbuf = comm.gather(fr_psd_1,root = 0)
                #PSD2_setbuf = comm.gather(fr_psd_2,root = 0)
                endtimes_buff = comm.gather(ctime_idx[0],root = 0)
                                                    
                params_buff = comm.gather(params1, root = 0)
                paramsl_buff = comm.gather(params2, root = 0)
                    
                    #print norms_buff
                    
                if myid == 0:
                    
                    # AVERAGE ONLY OVER GOOD SEGS!
                    
                    # mask = avoided_buff < 1.
                    
                    # try: PSD1_mean = np.mean(np.array(PSD1_setbuf)[mask], axis = 0)
                    # except ValueError: PSD1_mean = np.zeros_like(PSD1_totset[0])
                    # try: PSD2_mean = np.mean(np.array(PSD2_setbuf)[mask], axis = 0)
                    # except ValueError: PSD2_mean = np.zeros_like(PSD1_totset[0])
                    #
                    # PSD1_totset.append(PSD1_mean)
                    # PSD2_totset.append(PSD2_mean)
                    #
                    # avoided += np.sum(avoided_buff)
                    

                    endtimes.append(endtimes_buff)
                    params.append(params_buff)
                    paramsl.append(paramsl_buff)                        
                    
                    endtime = np.max(endtimes_buff)
                    
                    
                    if minute % (nproc*25) == 0: 
                        
                        print 'analysed:', minute, 'minutes'
                        np.savez('%s/PSDS_params%s_%s.npz' % (out_path, run_name, cnt), endtimes = endtimes, params = params, paramsl= paramsl, minute=minute)
                            
                            
                        cnt+=1
                        
            ctime_nproc = []
            strain1_nproc = []
            strain2_nproc = []
            
        idx_block += 1    
            
exit()