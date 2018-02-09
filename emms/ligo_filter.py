import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import matplotlib.mlab as mlab
#from sympy.ntheory import factorint

FACTOR_LIMIT = 100
def bestFFTlength(n):
    while n % 128 != 0:
        n-=1
    return n

    max_fac = 10*FACTOR_LIMIT
    while max_fac >= FACTOR_LIMIT:
        for d in range(2,FACTOR_LIMIT):
            if n % d == 0:
                max_fac = d
                break
        n -= 1
    return n

def whitenbp_notch(strain_in, fs=4096, dt=1./4096):
    strain = strain_in.copy()
    # do bandpass and notch filtering
    # get filter coefficients
    coefs = get_filter_coefs(fs,bandpass=False)

    # filter it:
    strain_bp = filter_data(strain,coefs)

    # number of sample for the fast fourier transform:
    NFFT = 1*fs
    Pxx, freqs = mlab.psd(strain_bp, Fs = fs, NFFT = NFFT)

    # We will use interpolations of the ASDs computed above for whitening:
    psd = interp1d(freqs, Pxx)

    #Should really use analytic fit to PSD now that it is notched

    # now whiten the data 
    strain_whiten = whiten(strain_bp,psd,dt)

    # bandpass filter parameters
    lowcut=20 #43
    highcut=300 #260
    order = 4

    # bandpass filter coefficients
    # do bandpass as last filter
    nyq = 0.5*fs
    low = lowcut / nyq
    high = highcut / nyq
    bb, ab = butter(order, [low, high], btype='band')
    strain_bp = filtfilt(bb, ab, strain_whiten)
    
    return strain_bp

# generate linear time-domain filter coefficients, common to both H1 and L1.
# First, define some functions:

# This function will generate digital filter coefficients for bandstops (notches).
# Understanding it requires some signal processing expertise, which we won't get into here.
def iir_bandstops(fstops, fs, order=4):
    """ellip notch filter
    fstops is a list of entries of the form [frequency (Hz), df, df2]                           
    where df is the pass width and df2 is the stop width (narrower                              
    than the pass width). Use caution if passing more than one freq at a time,                  
    because the filter response might behave in ways you don't expect.
    """
    nyq = 0.5 * fs

    # Zeros zd, poles pd, and gain kd for the digital filter
    zd = np.array([])
    pd = np.array([])
    kd = 1

    # Notches
    for fstopData in fstops:
        fstop = fstopData[0]
        df = fstopData[1]
        df2 = fstopData[2]
        low = (fstop - df) / nyq
        high = (fstop + df) / nyq
        low2 = (fstop - df2) / nyq
        high2 = (fstop + df2) / nyq
        z, p, k = iirdesign([low,high], [low2,high2], gpass=1, gstop=6,
                            ftype='ellip', output='zpk')
        zd = np.append(zd,z)
        pd = np.append(pd,p)

    # Set gain to one at 100 Hz...better not notch there                                        
    bPrelim,aPrelim = zpk2tf(zd, pd, 1)
    outFreq, outg0 = freqz(bPrelim, aPrelim, 100/nyq)

    # Return the numerator and denominator of the digital filter                                
    b,a = zpk2tf(zd,pd,k)
    return b, a

def get_filter_coefs(fs, bandpass=True):
    
    # assemble the filter b,a coefficients:
    coefs = []

    # bandpass filter parameters
    lowcut=20 #43
    highcut=300 #260
    order = 4

    # Frequencies of notches at known instrumental spectral line frequencies.
    # You can see these lines in the ASD above, so it is straightforward to make this list.
    notchesAbsolute = np.array(
        [14.0,34.70, 35.30, 35.90, 36.70, 37.30, 40.95, 60.00, 
         120.00, 179.99, 304.99, 331.49, 510.02, 1009.99])
    # exclude notch below lowcut
    notchesAbsolute = notchesAbsolute[notchesAbsolute > lowcut]

    # notch filter coefficients:
    for notchf in notchesAbsolute:                      
        bn, an = iir_bandstops(np.array([[notchf,1,0.1]]), fs, order=4)
        coefs.append((bn,an))

    # Manually do a wider notch filter around 510 Hz etc.          
    bn, an = iir_bandstops(np.array([[510,200,20]]), fs, order=4)
    coefs.append((bn, an))

    # also notch out the forest of lines around 331.5 Hz
    bn, an = iir_bandstops(np.array([[331.5,10,1]]), fs, order=4)
    coefs.append((bn, an))

    if bandpass:
        # bandpass filter coefficients
        # do bandpass as last filter
        nyq = 0.5*fs
        low = lowcut / nyq
        high = highcut / nyq
        bb, ab = butter(order, [low, high], btype='band')
        coefs.append((bb,ab))
    
    return coefs

# and then define the filter function:
def filter_data(data_in,coefs):
    data = data_in.copy()
    for coef in coefs:
        b,a = coef
        # filtfilt applies a linear filter twice, once forward and once backwards.
        # The combined filter has linear phase.
        data = filtfilt(b, a, data)
    return data

# function to whiten data
def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    Nt = bestFFTlength(Nt)
    freqs = np.fft.rfftfreq(Nt, dt)
    print 'whitening...', Nt
    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain[:Nt])
    white_hf = hf / (np.sqrt(interp_psd(freqs) /dt/2.))
    white_ht = np.fft.irfft(white_hf, n=Nt)
    print 'done whitening...'
    return white_ht
