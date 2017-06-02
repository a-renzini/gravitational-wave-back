import qpoint as qp
import numpy as np
import healpy as hp
#from spider_analysis.map import synfast, cartview
import pylab
import math
import cmath
import OverlapFunctsSrc as ofs
from scipy import integrate
import response_function as rf
import matplotlib.pyplot as plt

#consider making this also a function

nsd = 16
#print hp.nside2npix(nsd)
Q = qp.QMap(nside=nsd, pol=True, accuracy='low',
            fast_math=True, mean_aber=True)

num = 240
ctime = 1418662800.+ 360.*np.arange(num) #seconds #this is ctime[i] for n[i]

#fix the zenith quaternion for HANFORD
lat =46.455102
lon =-119.407445   

#fix the zenith quaternion for LIVINGSTON
latp =30.611701
lonp =-90.740123

HaHb_R = np.zeros_like(ctime)
HaHb_I = np.zeros_like(ctime)
pixn = np.arange(num)
map_R = np.zeros(hp.nside2npix(nsd),dtype=np.float64)
map_I = np.zeros(hp.nside2npix(nsd),dtype=np.float64)

def n_n_p_arrays():
    i = 0
    n=np.zeros((num,4))
    n_p=np.zeros((num,4))
    while i < num:

        n_mat = Q.azel2bore(0., 90., None, None, lon, lat, ctime[i]) #we'll use this to rotate the quatmap
        n[i] = np.array(n_mat[0]) #n is the fixed direction

        n_p_mat = Q.azel2bore(0., 90., None, None, lonp, latp, ctime[i]) #we'll use this to rotate the quatmap
        n_p[i] = np.array(n_p_mat[0]) 
    
    ## quick check 
    # n_vect = ofs.m(np.deg2rad(Q.quat2radecpa(n)[1]), np.deg2rad(Q.quat2radecpa(n)[0]))
    # n_p_vect = ofs.m(np.deg2rad(Q.quat2radecpa(n_p)[1]), np.deg2rad(Q.quat2radecpa(n_p)[0]))
    # print np.dot(n_vect,n_p_vect)
    ##
        i+=1
    return n, n_p

n = n_n_p_arrays()[0]
n_p = n_n_p_arrays()[1]

#BUILDING THE CORRELATION ARRAY

#i = 0
# while i < num:
#
#     HaHb_R[i], HaHb_I[i], pixn[i] = rf.HaHb_corr(n[i],n_p[i],1.,nsd)
#
#     map_R[pixn[i]] = HaHb_R[i]
#     map_I[pixn[i]] = HaHb_I[i]
#
#     i+=1
#
#print HaHb
# hp.mollview(map_R)
# plt.savefig('map8_R.pdf')
# hp.mollview(map_I)
# plt.savefig('map8_I.pdf')

