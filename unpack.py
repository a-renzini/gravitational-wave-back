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
from scipy.optimize import curve_fit
from numpy import cos,sin
from matplotlib import cm
import glob
import errno

import os
import sys

import re

def sorted_nicely( l ):
    """ Sorts the given iterable in the way that is expected.
 
    Required arguments:
    l -- The iterable to be sorted.
 
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

maptyp = sys.argv[1]

out_path = './fits_files'
Mpp_map_path = './Mpp_maps'
fits_path = './fits/*.fits'
fits_paths = glob.glob(fits_path)
fits_stds = []
conds = []
b_path = './b_pixes.npz'
check_path = './checkfiles/*.npz'
check_paths = glob.glob(check_path)

fits_paths = sorted_nicely(fits_paths)
check_paths = sorted_nicely(check_paths)

times = []
for i in range(len(fits_paths)):
    times.append( map(int, re.findall('\d+', fits_paths[i])))

times = np.array(times).flatten()
print times

planckmaps = ['planck_1', 'planck_2', 'planck_3', 'planck_poi','planck_5', 'planck_4']
#print fits_paths


    
for i in range(len(fits_paths)):
    map_in = hp.read_map(fits_paths[i])/1.e30
    
    # hp.mollview(S_p,norm = 'hist', cmap = jet)
    # plt.savefig('%s/S_p%s.pdf' % (out_path,counter))
    # plt.close('all')
    #
    # fig = plt.figure()
    
    fwhm = 5*np.pi/180.
    map_in = hp.ud_grade(map_in,nside_out = 64)
    map_in = hp.sphtfunc.smoothing(map_in,fwhm = fwhm)
    
    if maptyp in planckmaps:
        
        jet = cm.jet
        jet.set_under("w")
        hp.mollview(map_in,norm = 'hist', cmap = jet)
        plt.savefig('%s/fitsmap%s.pdf' % (out_path,i)  )
        plt.close('all')
        
    else:
        
        hp.mollview(map_in)
        plt.savefig('%s/fitsmap%s.pdf' % (out_path,i))
        plt.close('all')

    fits_stds.append(np.std(map_in))
    
    checkfile = np.load(check_paths[i])

    Mpprime =  checkfile['M_p_pp']
    Mpp_map = np.sqrt(np.diag(np.linalg.pinv(Mpprime,rcond=1.e-10)))
    norm = np.sqrt(np.sum((checkfile['A_pp'])))
    
    fwhm = 5*np.pi/180.
    Mpp_norm = hp.ud_grade(Mpp_map*norm,nside_out = 64)
    Mpp_norm = hp.sphtfunc.smoothing(Mpp_norm,fwhm = fwhm)

    hp.mollview( Mpp_norm)
    plt.savefig('%s/Mpp_map%s.pdf' % (Mpp_map_path,i))
    plt.close('all')
    print norm
    
    conds.append(np.linalg.cond(Mpprime))    
    
npix = len(map_in)
nside_out = hp.npix2nside(npix)
nside_in = 32

conds_poly = np.poly1d(np.polyfit(times, conds, 2))

plt.figure()
plt.plot(times, conds,label = 'conditions')
#plt.plot(times, conds_poly(times), 'r--',label = 'Polynomial fit')
plt.yscale('log')
plt.ylabel('matrix condition')
plt.xlabel('time analysed (mins)')
plt.legend()
plt.savefig('conds.pdf')

#stds_poly = np.poly1d(np.polyfit(times, fits_stds, 4))

plt.figure()
plt.plot(times, fits_stds, label = 'Standard deviation')
plt.yscale('log')
plt.ylabel('stdev')
plt.xlabel('time analysed (mins)')
plt.legend()
plt.savefig('fits_stds.pdf' )

np.savez('fits_stds_%s.npz' % maptyp, fits_stds = fits_stds, times = times )

b_file = np.load(b_path)

b_map = np.zeros(hp.nside2npix(nside_in))
b_map[b_file['b_pixes'].astype(int)] = 1.

fig = plt.figure()
hp.mollview(b_map)
#hp.mollview(np.zeros(hp.nside2npix(nside_in)))
#hp.visufunc.projscatter(hp.pix2ang(nside_in,b_file['b_pixes'].astype(int)))
plt.savefig('%s/b_pixes.pdf' % out_path)
plt.close('all')

np.savez('b_pixes_%s.npz' % maptyp, b_pixes = b_file['b_pixes'].astype(int) )

