import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.special import spherical_jn, sph_harm
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py
import datetime as dt
import pytz
import qpoint as qp
import healpy as hp

# LIGO-specific readligo.py 
import readligo as rl
import ligo_filter as lf
from gwpy.time import tconvert

def dfreq_factor(f,ell,b,alpha=3.,H0=68.0,f0=100.):
    # f : frequency (Hz)
    # ell : multipole
    # alpha: spectral index
    # b: baseline length (m)
    # f0: pivot frequency (Hz)
    # H0: Hubble rate today (km/s/Mpc

    km_mpc = 3.086e+19 # km/Mpc conversion
    c = 3.e8 # speed of light 
    #fac = 8.*np.pi**3/3./H0**2 * km_mpc**2 * f**3*(f/f0)**(alpha-3.) * spherical_jn(ell, 2.*np.pi*f*b/c)
    fac = f**3*(f/f0)**(alpha-3.) * spherical_jn(ell, 2.*np.pi*f*b/c)
    # add band pass and notches here
 
    return fac

def freq_factor(ell,b,fmin,fmax,alpha=3.,H0=68.0,f0=100.):
    return integrate.quad(dfreq_factor,fmin,fmax,args=(ell,b))[0]
    
def vec2azel(v1,v2):
    # calculate the viewing angle from location at v1 to v2
    # Cos(elevation+90) = (x*dx + y*dy + z*dz) / Sqrt((x^2+y^2+z^2)*(dx^2+dy^2+dz^2))
    # Cos(azimuth) = (-z*x*dx - z*y*dy + (x^2+y^2)*dz) / Sqrt((x^2+y^2)(x^2+y^2+z^2)(dx^2+dy^2+dz^2))
    # Sin(azimuth) = (-y*dx + x*dy) / Sqrt((x^2+y^2)(dx^2+dy^2+dz^2))
    
    v = v2-v1
    d = np.sqrt(np.dot(v,v))
    cos_el = np.dot(v2,v)/np.sqrt(np.dot(v2,v2)*np.dot(v,v))
    el = np.arccos(cos_el)-np.pi/2.
    cos_az = (-v2[2]*v2[0]*v[0] - v2[2]*v2[1]*v[1] + (v2[0]**2+v2[1]**2)*v[2])/np.sqrt((v2[0]**2+v2[1]**2)*np.dot(v2,v2)*np.dot(v,v))
    sin_az = (-v2[1]*v[0] + v2[0]*v[1])/np.sqrt((v2[0]**2+v2[1]**2)*np.dot(v,v))
    az = np.arctan2(sin_az,cos_az)

    return az, el, d

def midpoint(lat1,lon1,lat2,lon2):
    # http://www.movable-type.co.uk/scripts/latlong.html 
    Bx = np.cos(lat2) * np.cos(lon2-lon1)
    By = np.cos(lat2) * np.sin(lon2-lon1)

    latMid = np.arctan2(np.sin(lat1) + np.sin(lat2),np.sqrt((np.cos(lat1)+Bx)*(np.cos(lat1)+Bx) + By*By))
    lonMid = lon1 + np.arctan2(By, np.cos(lat1) + Bx)

    # bearing of great circle at mid point (azimuth wrt local North) 
    y = np.sin(lon2-lonMid) * np.cos(lat2);
    x = np.cos(latMid)*np.sin(lat2) - np.sin(latMid)*np.cos(lat2)*np.cos(lon2-lonMid);
    brng = np.degrees(np.arctan2(y, x));

    return latMid,lonMid, brng


# sampling rate:
fs = 4096
ligo_data_dir = '/Users/contaldi/Students/Arianna/LIGO/ligo_data/'
filelist = rl.FileList(directory=ligo_data_dir)

# Configuration: radians and metres, Earth-centered frame
H1_lon = -2.08405676917
H1_lat = 0.81079526383
H1_elev = 142.554
H1_vec = np.array([-2.16141492636e+06, -3.83469517889e+06, 4.60035022664e+06]) 

L1_lon = -1.58430937078
L1_lat = 0.53342313506
L1_elev = -6.574
L1_vec = np.array([-7.42760447238e+04, -5.49628371971e+06, 3.22425701744e+06])

V1_lon = 0.18333805213
V1_lat = 0.76151183984
V1_elev = 51.884
V1_vec = np.array([4.54637409900e+06, 8.42989697626e+05, 4.37857696241e+06])

# work out viewing angle of baseline H1->L1
az_b, el_b, baseline_length = vec2azel(H1_vec,L1_vec)
# position of mid point and angle of great circle connecting to observatories
latMid,lonMid, azMid = midpoint(H1_lat,H1_lon,L1_lat,L1_lon)

#print np.degrees(az_b), np.degrees(el_b), b_mag

#print freq_factor(16,baseline_length,0.01,300.)
#exit()

##############
# ARIANNA
# I need the gamma functions in the lab 'frame' here

# gammaI(nside) = 
#etc.
###############

# create qpoint object
nside = 64
data_map = np.zeros(hp.nside2npix(nside))
hits_map = np.zeros_like(data_map)
Q = qp.QPoint(accuracy='low', fast_math=True, mean_aber=True)#, num_threads=1)

# define start and stop time to search
# in GPS seconds
#start = 931035615 #931079472
#stop  = 971622015 #931086336
#start = 931079472
#stop  = 931086336
start = 931200000
stop = 931300000

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

# To-do: Add flagging of injections

# Now loop over all segments found
for sdx, (begin, end) in enumerate(zip(segs_begin,segs_end)):    
    
    strain_H1, meta_H1, dq_H1 = rl.getstrain(begin, end, 'H1', filelist=filelist)
    strain_L1, meta_L1, dq_L1 = rl.getstrain(begin, end, 'L1', filelist=filelist)
    strain_x = strain_H1*strain_L1
    
    # Figure out unix time for this segment
    # This is the ctime for qpoint
    utc_begin = tconvert(meta_H1['start']).replace(tzinfo=pytz.utc)
    utc_end = tconvert(meta_H1['stop']).replace(tzinfo=pytz.utc)
    ctime = np.arange((utc_begin - epoch).total_seconds(),(utc_end - epoch).total_seconds(),meta_H1['dt'])
    if len(ctime)/fs < 16: continue
    print utc_begin, utc_end

    if len(ctime)/fs > 120:
        # split segment into sub parts
        # not interested in freq < 40Hz
        n_split = np.int(len(ctime)/(60*fs))
        print 'split ', n_split, len(ctime)/fs
        ctime = np.array_split(ctime, n_split)
        strain_x = np.array_split(strain_x, n_split)
    else:
        # add dummy dimension
        n_split = 1
        ctime = ctime[None,...]
        strain_x = strain_x[None,...]
        
    for idx, (ct_split, s_split) in enumerate(zip(ctime, strain_x)):
        
        # get quaternions for H1->L1 baseline (use as boresight)
        q_b = Q.azel2bore(np.degrees(az_b), np.degrees(el_b), None, None, np.degrees(H1_lon), np.degrees(H1_lat), ct_split)
        # get quaternions for bisector pointing (use as boresight)
        q_n = Q.azel2bore(azMid, 90.0, None, None, np.degrees(lonMid), np.degrees(latMid), ct_split)
        
        pix_b, s2p, c2p = Q.quat2pix(q_b, nside=nside, pol=True) #spin-2

        print '{}/{} {} seconds'.format(idx,n_split,len(ct_split)/fs)
        
        # Filter the data
        #strain_H1 = lf.whitenbp_notch(strain_H1)
        #strain_L1= lf.whitenbp_notch(strain_L1)
        s_filt = lf.whitenbp_notch(s_split)
        #print 'filtered data...'
        # Downsample data from 4096 Hz to ?????
    
        # Do our mapping algorithm here
        # Accumulate data map and projection operator
        
        for p,s in zip(pix_b,s_filt):
            data_map[p] += s
            hits_map[p] += 1
        #print 'added map...'
    out = np.copy(data_map)
    out[hits_map > 0] /= hits_map[hits_map > 0]
    out[hits_map==0.] = hp.UNSEEN
    hp.mollview(out)
    plt.savefig('map.pdf')
    hp.write_map("map.fits",out)

data_map[hits_map > 0] /= hits_map[hits_map > 0]
data_map[hits_map == 0.] = hp.UNSEEN
hp.mollview(data_map)
plt.savefig('map.pdf')
hp.write_map("map.fits",data_map)
exit()

NFFT = 1*fs
fmin = 10
fmax = 2000
Pxx_H1, freqs = mlab.psd(strain_H1, Fs = fs, NFFT = NFFT)
Pxx_L1, freqs = mlab.psd(strain_L1, Fs = fs, NFFT = NFFT)
Pxx_x, freqs = mlab.psd(strain_x, Fs = fs, NFFT = NFFT)

# We will use interpolations of the ASDs computed above for whitening:

psd_H1 = interp1d(freqs, Pxx_H1)
psd_L1 = interp1d(freqs, Pxx_L1)
psd_x = interp1d(freqs, Pxx_x)

# plot the ASDs:
plt.figure()
plt.loglog(freqs, np.sqrt(Pxx_H1)/(sdx+1),'r',label='H1 strain')
plt.loglog(freqs, np.sqrt(Pxx_L1)/(sdx+1),'g',label='L1 strain')
plt.loglog(freqs, np.sqrt(Pxx_x)/(sdx+1),'b',label= 'L1xH1 strain')
#plt.axis([fmin, fmax, 1e-24, 1e-19])
plt.grid('on')
plt.ylabel('ASD (strain/rtHz)')
plt.xlabel('Freq (Hz)')
plt.legend(loc='best')
plt.title('ASDs L1 and H1')
plt.savefig('ASDs.pdf')

exit()

#----------------------------------------------------------------
# Load LIGO data from a single file
#----------------------------------------------------------------
# First from H1
fn_H1 = 'ligo_data'+'/'+'H-H1_LOSC_4_V1-1126259446-32.hdf5'
strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
# and then from L1
fn_L1 = 'ligo_data'+'/'+'L-L1_LOSC_4_V1-1126259446-32.hdf5'
strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')

# sampling rate:
fs = 4096
# both H1 and L1 will have the same time vector, so:
time = time_H1
# the time sample interval (uniformly sampled!)
dt = time[1] - time[0]


# read in the NR template
NRtime, NR_H1 = np.genfromtxt('ligo_data'+'/'+'GW150914_4_NR_waveform.txt').transpose()

# First, let's look at the data and print out some stuff:

# this doesn't seem to work for scientific notation:
# np.set_printoptions(precision=4)

print '  time_H1: len, min, mean, max = ', \
   len(time_H1), time_H1.min(), time_H1.mean(), time_H1.max()
print 'strain_H1: len, min, mean, max = ', \
   len(strain_H1), strain_H1.min(),strain_H1.mean(),strain_H1.max()
print 'strain_L1: len, min, mean, max = ', \
   len(strain_L1), strain_L1.min(),strain_L1.mean(),strain_L1.max()
    
#What's in chan_dict? See https://losc.ligo.org/archive/dataset/GW150914/
bits = chan_dict_H1['DATA']
print 'H1     DATA: len, min, mean, max = ', len(bits), bits.min(),bits.mean(),bits.max()
bits = chan_dict_H1['CBC_CAT1']
print 'H1 CBC_CAT1: len, min, mean, max = ', len(bits), bits.min(),bits.mean(),bits.max()
bits = chan_dict_H1['CBC_CAT2']
print 'H1 CBC_CAT2: len, min, mean, max = ', len(bits), bits.min(),bits.mean(),bits.max()
bits = chan_dict_L1['DATA']
print 'L1     DATA: len, min, mean, max = ', len(bits), bits.min(),bits.mean(),bits.max()
bits = chan_dict_L1['CBC_CAT1']
print 'L1 CBC_CAT1: len, min, mean, max = ', len(bits), bits.min(),bits.mean(),bits.max()
bits = chan_dict_L1['CBC_CAT2']
print 'L1 CBC_CAT2: len, min, mean, max = ', len(bits), bits.min(),bits.mean(),bits.max()
print 'In both H1 and L1, all 32 seconds of data are present (DATA=1), '
print "and all pass data quality (CBC_CAT1=1 and CBC_CAT2=1)."

# plot +- 5 seconds around the event:
tevent = 1126259462.422         # Mon Sep 14 09:50:45 GMT 2015 
deltat = 5.                     # seconds around the event
# index into the strain time series for this time interval:
indxt = np.where((time_H1 >= tevent-deltat) & (time_H1 < tevent+deltat))

plt.figure()
plt.plot(time_H1[indxt]-tevent,strain_H1[indxt],'r',label='H1 strain')
plt.plot(time_L1[indxt]-tevent,strain_L1[indxt],'g',label='L1 strain')
plt.xlabel('time (s) since '+str(tevent))
plt.ylabel('strain')
plt.legend(loc='lower right')
plt.title('Advanced LIGO strain data near GW150914')
plt.savefig('GW150914_strain.pdf')
# number of sample for the fast fourier transform:
NFFT = 1*fs
fmin = 10
fmax = 2000
Pxx_H1, freqs = mlab.psd(strain_H1, Fs = fs, NFFT = NFFT)
Pxx_L1, freqs = mlab.psd(strain_L1, Fs = fs, NFFT = NFFT)

# We will use interpolations of the ASDs computed above for whitening:
psd_H1 = interp1d(freqs, Pxx_H1)
psd_L1 = interp1d(freqs, Pxx_L1)

# plot the ASDs:
plt.figure()
plt.loglog(freqs, np.sqrt(Pxx_H1),'r',label='H1 strain')
plt.loglog(freqs, np.sqrt(Pxx_L1),'g',label='L1 strain')
plt.axis([fmin, fmax, 1e-24, 1e-19])
plt.grid('on')
plt.ylabel('ASD (strain/rtHz)')
plt.xlabel('Freq (Hz)')
plt.legend(loc='upper center')
plt.title('Advanced LIGO strain data near GW150914')
plt.savefig('GW150914_ASDs.pdf')

# now whiten the data from H1 and L1, and also the NR template:
strain_H1_whiten = lf.whiten(strain_H1,psd_H1,dt)
strain_L1_whiten = lf.whiten(strain_L1,psd_L1,dt)
NR_H1_whiten = lf.whiten(NR_H1,psd_H1,dt)

# do bandpass and notch filtering

# get filter coefficients
coefs = lf.get_filter_coefs(fs)

# filter it:
strain_H1_whitenbp = lf.filter_data(strain_H1_whiten,coefs)
strain_L1_whitenbp = lf.filter_data(strain_L1_whiten,coefs)
NR_H1_whitenbp = lf.filter_data(NR_H1_whiten,coefs)

# We need to suppress the high frequencies with some bandpassing:
#bb, ab = butter(4, [20.*2./fs, 300.*2./fs], btype='band')
#strain_H1_whitenbp = filtfilt(bb, ab, strain_H1_whiten)
#strain_L1_whitenbp = filtfilt(bb, ab, strain_L1_whiten)
#NR_H1_whitenbp = filtfilt(bb, ab, NR_H1_whiten)

# plot the data after whitening:
# first, shift L1 by 7 ms, and invert. See the GW150914 detection paper for why!
strain_L1_shift = -np.roll(strain_L1_whitenbp,int(0.007*fs))

plt.figure()
plt.plot(time-tevent,strain_H1_whitenbp,'r',label='H1 strain')
plt.plot(time-tevent,strain_L1_shift,'g',label='L1 strain')
plt.plot(NRtime+0.002,NR_H1_whitenbp,'k',label='matched NR waveform')
plt.xlim([-0.1,0.05])
plt.ylim([-4,4])
plt.xlabel('time (s) since '+str(tevent))
plt.ylabel('whitented strain')
plt.legend(loc='lower left')
plt.title('Advanced LIGO WHITENED strain data near GW150914')
plt.savefig('GW150914_strain_whitened.pdf')

Pxx_H1, freqs = mlab.psd(strain_H1_whitenbp, Fs = fs, NFFT = NFFT)
Pxx_L1, freqs = mlab.psd(strain_L1_whitenbp, Fs = fs, NFFT = NFFT)

# We will use interpolations of the ASDs computed above for whitening:
psd_H1 = interp1d(freqs, Pxx_H1)
psd_L1 = interp1d(freqs, Pxx_L1)

# plot the ASDs:
plt.figure()
plt.loglog(freqs, np.sqrt(Pxx_H1),'r',label='H1 strain')
plt.loglog(freqs, np.sqrt(Pxx_L1),'g',label='L1 strain')
#plt.axis([fmin, fmax, 1e-24, 1e-19])
plt.grid('on')
plt.ylabel('ASD (strain/rtHz)')
plt.xlabel('Freq (Hz)')
plt.legend(loc='upper center')
plt.title('Advanced LIGO strain data near GW150914')
plt.savefig('GW150914_ASDs_filt.pdf')

plt.figure()
plt.plot(time_H1[indxt]-tevent,strain_H1_whitenbp[indxt],'r',label='H1 strain')
plt.plot(time_L1[indxt]-tevent,strain_L1_whitenbp[indxt],'g',label='L1 strain')
plt.xlabel('time (s) since '+str(tevent))
plt.ylabel('strain')
plt.legend(loc='lower right')
plt.title('Advanced LIGO strain data near GW150914')
plt.savefig('GW150914_strain_filt.pdf')
