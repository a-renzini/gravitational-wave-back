import qpoint as qp
import numpy as np
import healpy as hp
#from spider_analysis.map import synfast, cartview
import pylab
import math
import cmath
import OverlapFunctsSrc as ofs
from scipy import integrate
import quat_rotation as qr
import quat_2_abg as q2abg
import matplotlib.pyplot as plt

# initialize, maybe change a few options from their defaults
nsd=32

Q = qp.QMap(nside=nsd, pol=True, accuracy='low',
            fast_math=True, mean_aber=True)

"""
                az         boresight azimuth (degrees)
                el         boresight elevation (degrees)
                pitch      boresight pitch (degrees); can be None
                roll       boresight pitch (degrees); can be None
                lon        observer longitude (degrees)
                lat        observer latitude (degrees)
                ctime      unix time in seconds UTC
                q          output quaternion array initialized by user
"""

##################### CREATE RANDOM HPLUS/CROSS FIELDS  ###########################

l=[1]*32
cls=[1]*32
i=0
while i<32:
    cls[i]=1./(i+1.)**2.
    i+=1

#ell, cls = cls[0], cls[1:]
hplus = np.vstack(hp.synfast(cls, nside=nsd, pol=True, new=True))
hcross = np.vstack(hp.synfast(cls, nside=nsd, pol=True, new=True))
lenmap = len(hplus)


map = np.zeros(lenmap,dtype=np.float64)
ip = hp.ang2pix(nsd,np.radians(90.),np.radians(0.))
map[ip]=10.
ip = hp.ang2pix(nsd,np.radians(80.),np.radians(15.))
map[ip]=15.
ip = hp.ang2pix(nsd,np.radians(90.),np.radians(0.))
map[ip]=20.
ip = hp.ang2pix(nsd,np.radians(100.),np.radians(365.))
map[ip]=25.
ip = hp.ang2pix(nsd,np.radians(90.),np.radians(0.))
map[ip]=30.
hp.mollview(map)
plt.savefig('map.pdf')

#np.savetxt('hplus.txt', hplus, delimiter=' ', newline='\n')
#np.savetxt('hcross.txt', hcross, delimiter=' ', newline='\n')

############## CREATE HPMAP OF QUATERNIONS #######################
#make the map of 0s

#row = np.array([0.]*4)
#quatmap = [row for _ in range(lenmap)]

dec_quatmap,ra_quatmap = hp.pix2ang(nsd,np.arange(lenmap)) #hp.interpolate
radec_quatmap = np.array([np.rad2deg(ra_quatmap),np.rad2deg(dec_quatmap)]).T
np.savetxt('radec_quatmap.txt', radec_quatmap, delimiter=' ', newline='\n')

fixed_ctime = 1418662800. #1418662800. #this will become ctime[i] for n[i]
lat =39.4
lon =-103.5
#
#fix the zenith quaternion
n = Q.azel2bore(0., 90., None, None, lon, lat, fixed_ctime) #we'll use this to rotate the quatmap
n = np.array(n[0]) #n is the fixed direction
print n
ran, decn, pan = np.array(Q.quat2radecpa(n))
print ran, decn, pan
#print q2abg.quat_2_abg(q)
print Q.radecpa2quat(ran, decn, pan)
#write the initial quatmap


quatmap = Q.radecpa2quat(np.rad2deg(ra_quatmap), np.rad2deg(dec_quatmap-np.pi*0.5), 0.*np.ones_like(ra_quatmap)) #smth weird is happening: ra should have range 0,2pi


np.savetxt('quatmap.txt', quatmap, delimiter=' ', newline='\n')
quatmap_rotated = np.ones_like(quatmap)
radecpamap_ini = Q.quat2radecpa(quatmap)
radecpamapnumpy_ini = np.array(radecpamap_ini).T
np.savetxt('radecpamap_ini.txt', radecpamapnumpy_ini, delimiter=' ', newline='\n')

#####################\SPINOFF

#pointing_vects_x, pointing_vects_y, pointing_vects_z = ofs.m(ra_quatmap,dec_quatmap-np.pi*0.5)
#pointing_vects = pointing_vects_x, pointing_vects_y, pointing_vects_z
#pointing_vects_T = np.array(pointing_vects).T
#np.savetxt('pointing_vects.txt', pointing_vects_T, delimiter=' ', newline='\n')
#
#quatized_vects = np.ones_like(quatmap)
#i = 0
#while i < lenmap:
#    quatized_vects[i] = [0.,pointing_vects_x[i], pointing_vects_y[i], pointing_vects_z[i]]
#    i+=1
#quatized_vects_T = np.array(quatized_vects).T
#np.savetxt('quatized_vects.txt', quatized_vects, delimiter=' ', newline='\n')
#
#quatmap_rotated_SO = np.ones_like(quatmap)
#i = 0
#while i < lenmap:
#    quatmap_rotated_SO[i] = qr.quat_rot(n,quatized_vects[i])
#    i+=1
#np.savetxt('quatmap_rotated_SO.txt', quatmap_rotated_SO, delimiter=' ', newline='\n')
#
#radecpamap_SO = Q.quat2radecpa(- quatmap_rotated_SO)
#radecpamapnumpy_SO = np.array(radecpamap_SO).T
#np.savetxt('radecpamap_SO.txt', radecpamapnumpy_SO, delimiter=' ', newline='\n')

#####################/SPINOFF

i = 0
while i < lenmap:
    quatmap_rotated[i] = qr.quat_mult(n,quatmap[i])
    i+=1
print '++++++'
rot_ip = Q.quat2pix(quatmap_rotated,nsd)[0]
print rot_ip[0]
map_rot = np.zeros_like(map)
map_rot = map[rot_ip]
hp.mollview(map_rot)
plt.savefig('map_rot.pdf')

theta, phi = hp.pix2ang(nsd,np.arange(lenmap)) 
fplush_map = ofs.FplusH(theta,phi)



vec_north = hp.ang2vec( 0.,0.)
vec_m = np.array(hp.pix2vec(nsd,np.arange(lenmap)))
print vec_m
print vec_m.shape
print vec_north
coseta = vec_m[2,:]
print coseta

exit()
np.savetxt('quatmap_rotated.txt', quatmap_rotated, delimiter=' ', newline='\n') #quatmap[i] is the quaternion corresponding to initial pixel i

radecpamap = Q.quat2radecpa(- quatmap_rotated)
radecpamapnumpy = np.array(radecpamap).T
np.savetxt('radecpamap.txt', radecpamapnumpy, delimiter=' ', newline='\n')
