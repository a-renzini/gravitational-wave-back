import qpoint as qp
import numpy as np
import healpy as hp
import pylab
import math
import cmath
import OverlapFunctsSrc as ofs
from scipy import integrate
from response_function import rotation_pix
from response_function import alpha
from quat_rotation import quat_inv
import matplotlib.pyplot as plt
import stokes_map as sm

#set the frequency:

f = 1   #Hz

#set the maps:

nsd = 32
Q = qp.QMap(nside=nsd, pol=True, accuracy='low',
            fast_math=True, mean_aber=True)

lenmap =  hp.nside2npix(nsd)
theta, phi = hp.pix2ang(nsd,np.arange(lenmap)) 
m_array = np.arange(lenmap)
        
#Initial gammas, in the 'great circle' frame
gammaI = ofs.gammaIHL(theta,phi)

from camb.bispectrum import threej

def ThreeJC_2_camb(l2i, m2i, l3i, m3i):
    arr = threej(l2i, l3i, m2i, m3i)
    lmin = np.max([np.abs(l2i - l3i), np.abs(m2i + m3i)])
    print lmin
    lmax = l2i + l3i
    fj = np.zeros(lmax + 2, dtype=arr.dtype)
    fj[lmin:lmax + 1] = arr
    return fj, lmin, lmax
 
a = ThreeJC_2_camb(20,1,1,1)
print a
exit()

print hp.map2alm(gammaI)#[, lmax, mmax, iter, pol, ...])

