import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py
import datetime as dt
import pytz
from scipy import interpolate, signal

# LIGO-specific readligo.py 
import readligo as rl
import ligo_filter as lf
from gwpy.time import tconvert
# sampling rate:
fs = 4096
del_t = 1./fs
ligo_data_dir = '/Users/pai/Data/'
filelist = rl.FileList(directory=ligo_data_dir)
# define start and stop time to search
# in GPS seconds
start =  931217408 #931079472
stop  =  931291136 #931086336
#start = 931079472
#stop  = 931086336
#start = 931200000
#stop = 931300000

# convert LIGO GPS time to datetime
# make sure datetime object knows it is UTC timezone
utc_start = tconvert(start).replace(tzinfo=pytz.utc)
utc_stop = tconvert(stop).replace(tzinfo=pytz.utc)

# 1970-1-1 in UTC defines epoch of unix time 
epoch = dt.datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)

print (utc_start - epoch).total_seconds()
print (utc_stop - epoch).total_seconds()

# get segments with required flag level
segs_H1 = rl.getsegs(start, stop, 'H1',flag='STOCH_CAT1', filelist=filelist)
good_data_H1 = np.zeros(stop-start,dtype=np.bool)
for (begin, end) in segs_H1:
    good_data_H1[begin-start:end-start] = True
    
segs_L1 = rl.getsegs(start, stop, 'L1',flag='STOCH_CAT1', filelist=filelist)
good_data_L1 = np.zeros(stop-start,dtype=np.bool)
for (begin, end) in segs_L1:
    good_data_L1[begin-start:end-start] = True

# add time bit at beginning and end to _AND_ of two timeseries
good_data = np.append(np.append(False,good_data_H1 & good_data_L1),False)
# do diff to identify segments
diff = np.diff(good_data.astype(int))
segs_begin = np.where(diff>0)[0] + start #+1
segs_end =  np.where(diff<0)[0] + start #+1

# re-define without first and last time bit
# This mask now defines conincident data from both L1 and H1
good_data = good_data_H1 & good_data_L1
strain_H1, meta_H1, dq_H1 = rl.getstrain(segs_begin[1], segs_end[1], 'H1', filelist=filelist)
#Get fast length
len_fft = 2**(np.int(np.log(len(strain_H1))/np.log(2))-4)
strain_H1 = strain_H1[0:len_fft]
time = np.arange(len(strain_H1))*del_t
#strain_H1[:cut] *= 0.
#strain_H1[-cut:] *= 0.
strain_H1_nowin = np.copy(strain_H1)
strain_H1_nowin *= signal.tukey(len_fft,alpha=0.05)
strain_H1 *= np.blackman(len_fft)
print np.std(strain_H1), np.std(strain_H1_nowin)
#plt.figure(figsize(10, 6))

#FFTleave out dt so spectra are not in 
#correct
sfft = np.fft.rfft(strain_H1,n=len_fft*2)
sfft_nowin = np.fft.rfft(strain_H1_nowin,n=len_fft*2)
#frequencies
freq = np.fft.rfftfreq(len_fft*2,d=del_t)

#cut out pad frequencies
sfft = sfft[:len_fft/2+1]
sfft_nowin = sfft_nowin[:len_fft/2+1]
freq = freq[:len_fft/2+1]

#define PSD
#include dt factor here to get correct units
psd = np.abs(sfft*np.conj(sfft)*del_t**2)
psd_nowin = np.abs(sfft_nowin*np.conj(sfft_nowin)*del_t**2)
#mlab.psd outputs psd in correct units given sampling frequency
Pxx, freq_welch = mlab.psd(strain_H1, Fs=fs, NFFT=20*fs,noverlap=fs,window=np.blackman(20*fs),scale_by_freq=True)
psd_welch = interpolate.interp1d(freq_welch, Pxx) 

#define bandpass
hpass=0.00001#50.
lpass=100000#350.
mask = (freq>hpass) & (freq < lpass)
mask_welch = (freq_welch>hpass) & (freq_welch < lpass)
sfft[~mask]*=0.
sfft_nowin[~mask] *=0.
psd[~mask]*=0.
dnu = freq[1]-freq[0]
#check that integral of PSD gives the same as std dev (or variance)
#this norm is borken
print np.sqrt(np.sum(psd[mask])*dnu),np.sqrt(np.sum(Pxx[mask_welch])*dnu)
plt.plot(freq[mask],np.sqrt(psd[mask]),label=r"PSD Blackman")
plt.plot(freq[mask],np.sqrt(psd_nowin[mask]),label=r"PSD Tukey Padded", alpha=0.5)
plt.plot(freq_welch[mask_welch],np.sqrt(Pxx[mask_welch]),label=r"Welch PSD Hanning")
#psd_model = np.zeros_like(freq)
#psd_model[1:] = (6.e-18*freq[1:]**-2 + 1.e-26*freq[1:]**1.7)**2
#psd_model[~mask] *=0.
#plt.plot(freq[mask],np.sqrt(psd_model[mask]),label=r"PSD Model")

plt.xlabel(r"$Hz$")
plt.ylabel(r"$h/\sqrt{Hz}$")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('carlospsds.png')
#psd = psd_model
#generate realization of complex spectra with correct normalisation
#to recover correct normalization after irfft
psd_model = psd*fs**2
psd_model_nowin = psd_nowin*fs**2
sim = (np.random.normal(0, 1., len_fft/2+1) + 1j*np.random.normal(0, 1., len_fft/2+1))*np.sqrt(psd_model/2.)
#set monopole to zero so force mean zero
sim[0] *= 0.
#psd of sim in physical units
psd_sim = np.abs(sim*np.conj(sim)*del_t**2)
plt.plot(freq[mask],np.sqrt(psd[mask]),label=r"PSD Blackman")
plt.plot(freq_welch[mask_welch],np.sqrt(Pxx[mask_welch]),label=r"Welch PSD Hanning")
plt.plot(freq,np.sqrt(psd_sim),label=r"Sim PSD",alpha=0.5)
#psd_model = np.zeros_like(freq)
#psd_model[1:] = (6.e-18*freq[1:]**-2 + 1.e-26*freq[1:]**1.7)**2
#psd_model[~mask] *=0.
#plt.plot(freq[mask],np.sqrt(psd_model[mask]),label=r"PSD Model")

plt.xlabel(r"$Hz$")
plt.ylabel(r"$h/\sqrt{Hz}$")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()

colored_sim = np.copy(sim)

#wfft = sfft/np.sqrt(np.abs(sfft)/del_t)
#white = np.fft.irfft(wfft)
colored_sim = np.fft.irfft(colored_sim)
strain_H1_rec = np.fft.irfft(sfft)
print np.std(colored_sim),np.std(strain_H1_rec)
plt.plot(time,colored_sim,label=r"Simulated Strain")
plt.plot(time,strain_H1_nowin,label=r"Strain",alpha=0.5)

plt.xlabel(r"$t$")
plt.ylabel(r"$h$")
plt.legend()
plt.tight_layout()
#or whiten data
white = np.zeros_like(sfft_nowin)
white_sim = np.zeros_like(white)
white[mask] = sfft_nowin[mask]/np.sqrt(psd_welch(freq)[mask])*np.sqrt(len_fft)/fs
white[~mask] *= 0.
white = np.fft.irfft(white)

white_sim[mask] = sim[mask]/np.sqrt(psd_model[mask])*np.sqrt(len_fft)
white_sim = np.fft.irfft(white_sim)
print np.std(white), np.std(white_sim)

plt.plot(time,white_sim,label=r"Simulated White Strain H1")
plt.plot(time,white,label=r"White Strain H1",alpha=0.5)

plt.xlabel(r"$t$")
plt.ylabel(r"$h$")
#plt.ylim([-5,5])
plt.legend()
plt.savefig('carlos.png')