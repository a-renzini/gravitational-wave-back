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
import MapBack as mb  #################

fs = 4096
nside_in = 16
nside_out = 16
lmax = 2
sim = True  
simtyp = 'mono'

#INTEGRATING FREQS:                                                                                                           
low_f = 80.
high_f = 300.

    
#DETECTORS
dects = ['H1','L1','V1']
ndet = len(dects)
nbase = int(ndet*(ndet-1)/2)
 
#create object of class:
run = mb.Telescope(nside_in,nside_out,lmax, fs, low_f, high_f, dects)

gammaI = run.gammaI[0]
hp.mollview(gammaI)
plt.savefig('gammaI.pdf')

lon = 90.
lat = 0.
ctime = 10000

Q = qp.QPoint(accuracy='low', fast_math=True, mean_aber=True)#, num_threads=1)


q = Q.rotate_quat(Q.azel2bore(0., 90.0, None, None, lon, lat, ctime)[0])
print q

npix = hp.nside2npix(nside_out)
print npix

rot_array = run.rotation_pix(np.arange(npix), q)  
gammaI_rot = gammaI[rot_array]

hp.mollview(gammaI_rot)
plt.savefig('gammaI_rot.pdf')

exit()