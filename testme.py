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

file = np.load('strain_h1.npz')

h1 = file['strain']

Nt = len(h1)
dt = 1./4096
low_f = 30.
high_f = 500.

freqs = np.fft.rfftfreq(2*Nt, dt)
freqs = freqs[:Nt/2+1]

hf = np.fft.rfft(h1, n=Nt, norm = 'ortho') 

print hf

Pxx, frexx = mlab.psd(hf, Fs=4096, NFFT=2*4096,noverlap=4096/2,window=np.blackman(2*4096),scale_by_freq=False)

hf_psd = interp1d(frexx,Pxx)
hf_psd_data = abs(hf.copy()*np.conj(hf.copy())) 

mask = (freqs>low_f) & (freqs < high_f)
mask2 = (freqs>80.) & (freqs < 300.)

norm = np.mean(hf_psd_data[mask])/np.mean(hf_psd(freqs)[mask])
norm2 = np.mean(hf_psd_data[mask2])/np.mean(hf_psd(freqs)[mask2])

np.savez('hf.npz', hf = hf,Pxx = Pxx, frexx = frexx, norm = norm, norm2 = norm2)

exit()