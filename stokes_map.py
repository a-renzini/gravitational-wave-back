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

nsd = 32
Q = qp.QMap(nside=nsd, pol=True, accuracy='low',
            fast_math=True, mean_aber=True)

num = 100
ctime = 1418662800.+ np.arange(num) #seconds #this will become ctime[i] for n[i]

#fix the zenith quaternion for HANFORD
lat =46.455102
lon =-119.407445   

#fix the zenith quaternion for LIVINGSTON
latp =30.611701
lonp =-90.740123

HaHb = np.zeros_like(ctime)

i = 0
while i < num:

    n = Q.azel2bore(0., 90., None, None, lon, lat, ctime[i]) #we'll use this to rotate the quatmap
    n = np.array(n[0]) #n is the fixed direction

    n_p = Q.azel2bore(0., 90., None, None, lonp, latp, ctime[i]) #we'll use this to rotate the quatmap
    n_p = np.array(n_p[0]) #n is the fixed direction
    
    HaHb[i] = rf. HaHb_corr(n,n_p,1.,32)
    
    i+=1

print HaHb