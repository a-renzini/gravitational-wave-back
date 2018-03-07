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
import stokefields as sfs
from numpy import cos,sin

# LIGO-specific readligo.py 
import readligo as rl
import ligo_filter as lf
from gwpy.time import tconvert



t_stream  = np.random.random_sample(10000)#/np.sqrt(2)+ 1.j* np.random.random_sample(10000)/np.sqrt(2)
print np.average(t_stream), np.std(t_stream)

f_stream = np.fft.rfft(t_stream ,norm = 'ortho')
freqs = np.arange(len(f_stream))

mask = (freqs>50) & (freqs < 150)
print len(f_stream)
f_stream = f_stream[mask]
print len(f_stream)
t_stream_tilda = np.fft.irfft(f_stream ,norm = 'ortho')

print np.average(t_stream), np.std(t_stream)

exit()