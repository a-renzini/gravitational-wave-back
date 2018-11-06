import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.special import spherical_jn, sph_harm    
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz, hanning
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
#import h5py
#import datetime as dt
#import pytz
#import pylab
#import qpoint as qp
#import healpy as hp
#from camb.bispectrum import threej
#import quat_rotation as qr
#from scipy.optimize import curve_fit
#import OverlapFunctsSrc as ofs
#from numpy import cos,sin
#from matplotlib import cm

file = np.load('problematic.npz')

h1 = file['h1']
Nt = len(h1)
h1 *= signal.tukey(Nt,alpha=0.05)

dl = 1./4096
low_f = 30.
high_f = 500.

freqs = np.fft.rfftfreq(Nt, dl)
freqs = freqs[:Nt]

hf = np.fft.rfft(h1, n=Nt, norm = 'ortho') 

hf = hf[:Nt]

print hf

Pxx, frexx = mlab.psd(h1, Fs=4096, NFFT=2*4096,noverlap=4096/2,window=np.blackman(2*4096),scale_by_freq=False)

hf_psd = interp1d(frexx,Pxx)
hf_psd_data = abs(hf.copy()*np.conj(hf.copy())) 

mask = (freqs>low_f) & (freqs < high_f)
mask2 = (freqs>80.) & (freqs < 300.)

#plt.figure()
#plt.loglog(hf_psd_data[mask])
#plt.loglog(hf_psd(freqs)[mask])
#plt.savefig('compare.pdf')


norm = np.mean(hf_psd_data[mask])/np.mean(hf_psd(freqs)[mask])
norm2 = np.mean(hf_psd_data[mask2])/np.mean(hf_psd(freqs)[mask2])

print norm, norm2

np.savez('hf.npz', hf = hf,Pxx = Pxx, frexx = frexx, norm = norm, norm2 = norm2)

exit()
