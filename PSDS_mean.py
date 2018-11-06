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


def PDX(frexx,a,b,c):
    #b = 1.
    #c = 1.
     
    return (a*1.e-22*((18.*b)/(0.1+frexx))**2)**2+(0.07*a*1.e-22)**2+((frexx/(2000.*c))*.4*a*1.e-22)**2

def notches():
    
    notch_fs = np.array([ 34.70, 35.30,35.90, 36.70, 37.30, 40.95, 60.00, 120.00, 179.99, 304.99, 331.9, 500.02,  1009.99])
    return notch_fs

def sigmas():

    sigma_fs = np.array([.5,.5,.5,.5,.5,.5,.5,1.,1.,1.,1.,5.,5.,1.])            
    return sigma_fs

def Pdx_notcher(freqx,Pdx):
    mask = np.ones_like(freqx, dtype = bool)

    for (idx_f,f) in enumerate(freqx):
        for i in range(len(notches())):
            if f > (notches()[i]-15.*sigmas()[i]) and f < (notches()[i]+15.*sigmas()[i]):
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

#FROM THE SHELL: data path, output path, type of input map, SNR level (noise =0, high, med, low)

data_path = sys.argv[1]
out_path =  sys.argv[2]
maptyp = '1pole'
noise_lvl = 1
noise_lvl = int(noise_lvl)
this_path = out_path

FULL_DESC = True

# poisson masked "flickering" map

poi = False

# if declared from shell, load checkpoint file 

try:
    sys.argv[3]
except (NameError, IndexError):
    checkpoint = False
else:
    checkpoint = True
    checkfile_path = sys.argv[3]

    
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

    PSD1_totset = []
    PSD2_totset = []
    
    if FULL_DESC == True:
        
        params = []
        norms = []
        endtimes = []
        
    minute = 0

else:
    
    PSD1_totset = None
    PSD2_totset = None
    
    if FULL_DESC == True:
        
        PSD_params = None
        norms = None

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
                                                                                                          
low_f = 80.
high_f = 300.

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
start = 1126224017  #start = start time of O1 ...      
stop  = 1137254417  #O1 end GPS     


##########################################################################

########################### data  massage  #################################

if myid == 0:
    print 'flagging'
    segs_begin, segs_end = run.flagger(start,stop,filelist)
    segs_begin = list(segs_begin)
    segs_end = list(segs_end)
    print segs_begin


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


            Nt = len(strain1)
            Nt = lf.bestFFTlength(Nt)

            freqs = np.fft.rfftfreq(2*Nt, 1./fs)
            freqs = freqs[:Nt/2+1]


            # frequency mask

            mask = (freqs>low_f) & (freqs < high_f)


            # repackage the strains & copy them (fool-proof); create empty array for the filtered, FFTed, correlated data

            strains = (strain1,strain2)
            strains_copy = (strain1.copy(),strain2.copy()) #calcualte psds from these



            ######################


            strain_in_1 = strains[0] 

            fs=4096       
            dt=1./fs

            '''WINDOWING & RFFTING.'''

            Nt = len(strain_in_1)
            Nt = lf.bestFFTlength(Nt)


            strain_in = strain_in_1[:Nt]
            strain_in_cp = np.copy(strain_in)
            strain_in_nowin = np.copy(strain_in)
            strain_in_nowin *= signal.tukey(Nt,alpha=0.05)
            strain_in_cp *= signal.tukey(Nt,alpha=0.05)

            freqs = np.fft.rfftfreq(2*Nt, dt)
            freqshal = np.fft.rfftfreq(Nt, dt)
            hf_halin = np.fft.rfft(strain_in_cp, n=Nt, norm = 'ortho') 
            hf_nowin = np.fft.rfft(strain_in_nowin, n=2*Nt, norm = 'ortho') #####!HERE! 03/03/18 #####

            # print 'lens', len(hf_halin), len(hf_nowin)
            # print 'means', np.mean(hf_halin), np.mean(hf_nowin)
            # print 'lens', len(hf_halin), len(hf_nowin)
            # print 'freqs', freqshal[-1], freqs[-1]
            # print 'means', np.mean(hf_halin), np.mean(hf_nowin)

            hf_nowin = hf_nowin[:Nt/2+1]
            freqs = freqs[:Nt/2+1]




            fstar = fs

            Pxx, frexx = mlab.psd(strain_in_nowin, Fs=fs, NFFT=2*fstar,noverlap=fstar/2,window=np.blackman(2*fstar),scale_by_freq=False)
            hf_psd = interp1d(frexx,Pxx)
            hf_psd_data = abs(hf_nowin.copy()*np.conj(hf_nowin.copy())) 


            # plt.figure()
            #
            # plt.loglog(freqs, np.abs(hf_nowin)**2, label = 'nowin PSD')
            # plt.loglog(freqshal, np.abs(hf_halin)**2, label = 'halin PSD')
            # plt.loglog(freqs, hf_psd(freqs)*2000., label = 'mlab PSD')
            # plt.loglog(freqs, hf_psd(freqs)*4000., label = 'mlab PSD')
            # plt.loglog(freqs, hf_psd(freqs)*1., label = 'mlab PSD')
            #
            # #plt.loglog(freqs, hf_psd(freqs)*4000., label = 'mlab PSD')
            #
            # #plt.loglog(self.PDX(frexx,a,b,c)[mask], label = 'notched pdx fit')
            # plt.legend()
            # plt.savefig('hfs.png' )



            mask = (freqs>low_f) & (freqs < high_f)
            masxx = (frexx>low_f) & (frexx < high_f)

            frexx_cp = np.copy(frexx)
            Pxx_cp = np.copy(Pxx)
            frexx_cp = frexx_cp[masxx]
            Pxx_cp = Pxx_cp[masxx]
            frexx_notch,Pxx_notch = Pdx_notcher(frexx_cp,Pxx_cp)
            frexcp = np.copy(frexx_notch)
            Pxcp = np.copy(Pxx_notch)


            try:
                fit = curve_fit(PDX, frexcp, Pxcp)#, bounds = ([0.,0.,0.],[2.,2.,2.])) 
                psd_params = fit[0]

            except RuntimeError:
                print("Error - curve_fit failed")
                psd_params = [10.,10.,10.]


            a,b,c = psd_params

            #print 'min:', minute, 'params:', psd_params

            min = 0.1
            max = 1.9

            norm = np.mean(hf_psd_data[mask])/np.mean(hf_psd(freqs)[mask])#/np.mean(self.PDX(freqs,a,b,c))
            print psd_params
            print 'norm: ' , norm

            #print 'norm: ' , norm

            #psd_params[0] = psd_params[0] 
    
            flag1 = False
    
            if a < min or a > (max/2*1.5): flag1= True
            if b < 2*min or b > 2*max: flag1 = True
            if c < 2*min or c > 12000*max: flag1 = True  # not drammatic if fit returns very high knee freq, ala the offset is ~1

            if norm > 3000. : flag1 = True

            #if a < min or a > (max): flags[idx_str] = True
            #if c < 2*min or c > 2*max: flags[idx_str] = True  # not drammatic if fit returns very high knee freq, ala the offset is ~1

    
            if flag1 == True: print 'bad segment!  params', a,b,c, 'ctime', ctime_idx[0]
    
            if flag1 == False:
        
                fr_psd_1 = Pdx_nanner(frexx_cp,hf_psd(frexx_cp))
                fr_psd_1 = fr_psd_1[1]*norm
                norm1 = norm
                params1 = psd_params

            strain_in_2 = strains[1]
    
            fs=4096       
            dt=1./fs

            '''WINDOWING & RFFTING.'''

            Nt = len(strain_in_2)
            Nt = lf.bestFFTlength(Nt)



            strain_in_2 = strain_in_2[:Nt]
            strain_in_cp_2 = np.copy(strain_in_2)
            strain_in_nowin_2 = np.copy(strain_in_2)
            strain_in_nowin_2 *= signal.tukey(Nt,alpha=0.05)
            strain_in_cp_2 *= signal.tukey(Nt,alpha=0.05)


            freqs = np.fft.rfftfreq(2*Nt, dt)
            freqshal = np.fft.rfftfreq(Nt, dt)
            hf_halin_2 = np.fft.rfft(strain_in_cp_2, n=Nt, norm = 'ortho') 
            hf_nowin_2 = np.fft.rfft(strain_in_nowin_2, n=2*Nt, norm = 'ortho') #####!HERE! 03/03/18 #####

            # print 'lens', len(hf_halin), len(hf_nowin)
            # print 'means', np.mean(hf_halin), np.mean(hf_nowin)
            # print 'lens', len(hf_halin), len(hf_nowin)
            # print 'freqs', freqshal[-1], freqs[-1]
            # print 'means', np.mean(hf_halin), np.mean(hf_nowin)

            hf_nowin_2 = hf_nowin_2[:Nt/2+1]
            freqs = freqs[:Nt/2+1]




            fstar = fs

            Pxx, frexx = mlab.psd(strain_in_nowin_2, Fs=fs, NFFT=2*fstar,noverlap=fstar/2,window=np.blackman(2*fstar),scale_by_freq=False)
            hf_psd = interp1d(frexx,Pxx)
            hf_psd_data_2 = abs(hf_nowin_2.copy()*np.conj(hf_nowin_2.copy())) 



            mask = (freqs>low_f) & (freqs < high_f)
            masxx = (frexx>low_f) & (frexx < high_f)

            frexx_cp = np.copy(frexx)
            Pxx_cp = np.copy(Pxx)
            frexx_cp = frexx_cp[masxx]

    
            Pxx_cp = Pxx_cp[masxx]
            frexx_notch,Pxx_notch = Pdx_notcher(frexx_cp,Pxx_cp)
            frexcp = np.copy(frexx_notch)
            Pxcp = np.copy(Pxx_notch)


            try:
                fit = curve_fit(PDX, frexcp, Pxcp)#, bounds = ([0.,0.,0.],[2.,2.,2.])) 
                psd_params = fit[0]

            except RuntimeError:
                print("Error - curve_fit failed")
                psd_params = [10.,10.,10.]


            a,b,c = psd_params
    
            #print 'min:', minute, 'params:', psd_params
    
            min = 0.1
            max = 1.9

            norm = np.mean(hf_psd_data_2[mask])/np.mean(hf_psd(freqs)[mask])#/np.mean(self.PDX(freqs,a,b,c))
            
            psd_params[0] = psd_params[0]*np.sqrt(norm) 
    
            flag2 = False
    
            if a < min or a > (max/2*1.5): flag2= True
            if b < 2*min or b > 2*max: flag2 = True
            if c < 2*min or c > 12000*max: flag2 = True  # not drammatic if fit returns very high knee freq, ala the offset is ~1
    
            if norm > 3000. : flag2 = True

        
    
            if flag2 == True or flag1 == True:
                if flag1 == True: print 'there was a badseg in H' 
                else: print 'bad segment!  params', psd_params, 'ctime', ctime_idx[0]
    
            else:
                    
                fr_psd_2 = Pdx_nanner(frexx_cp,hf_psd(frexx_cp))  
                fr_psd_2 = fr_psd_2[1]*norm          
                norm2 = norm
                params2 = psd_params
            #print 'analysed:', minute, 'minutes'
            
            if myid == 0:
                
                PSD1_setbuf = nproc * [np.zeros_like(fr_psd_2)]
                PSD2_setbuf = nproc * [np.zeros_like(fr_psd_2)]
                endtimes_buff = nproc *[0]
                endtime = 0                
                
                if FULL_DESC == True:
                    
                    norms_buff = nproc *[0]
                    params_buff = nproc * [np.zeros_like(psd_params)]
                
                minute += nproc
                
            else: 
                
                PSD1_setbuf = None
                PSD2_setbuf = None
                endtimes_buff = None
                endtime = None   
                
                if FULL_DESC == True:
                    
                    norms_buff = None
                    params_buff = None 
            
            if ISMPI: 
                                
                comm.barrier()
                
                PSD1_setbuf = comm.gather(fr_psd_1,root = 0)
                PSD2_setbuf = comm.gather(fr_psd_2,root = 0)
                endtimes_buff = comm.gather(ctime_idx[0],root = 0)
                
                if FULL_DESC == True:
                    
                    norms_buff = comm.gather(norm1, root = 0)
                    params_buff = comm.gather(params1, root = 0)
                
                if myid == 0:
                    
                    PSD1_mean = np.mean(PSD1_setbuf, axis = 0)
                    PSD2_mean = np.mean(PSD2_setbuf, axis = 0)
        
                    PSD1_totset.append(PSD1_mean)            
                    PSD2_totset.append(PSD2_mean)                                    
                    
                    if FULL_DESC == True:
                        
                        endtimes.append(endtimes_buff)
                        norms.append(norms_buff)
                        params.append(params_buff)
                        
                    
                    endtime = np.max(endtimes_buff)
                    
                    
                    if minute % (nproc*15) == 0: 
                        
                        if FULL_DESC == False:
                            
                            print 'analysed:', minute, 'minutes'
                            np.savez('%s/PSDS_meaned.npz' % out_path, PSD1_totset =PSD1_totset, PSD2_totset = PSD2_totset, ctime_end = endtime)
                        
                        if FULL_DESC == True:
                            print 'analysed:', minute, 'minutes'
                            np.savez('%s/PSDS_meaned.npz' % out_path, PSD1_totset =PSD1_totset, PSD2_totset = PSD2_totset, endtimes = endtimes, params = params, norms = norms)
                            
            ctime_nproc = []
            strain1_nproc = []
            strain2_nproc = []
            
        idx_block += 1    
            
exit()
