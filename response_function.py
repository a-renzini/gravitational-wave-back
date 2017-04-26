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
nsd=8
Q = qp.QMap(nside=nsd, pol=True, accuracy='low',
            fast_math=True, mean_aber=True)
            
R_earth=6378137  #radius of the earth in metres

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

l=[1]*nsd
cls=[1]*nsd
i=0
while i<nsd:
    cls[i]=1./(i+1.)**2.
    i+=1

#ell, cls = cls[0], cls[1:]
hplus = np.vstack(hp.synfast(cls, nside=nsd, pol=True, new=True))
hcross = np.vstack(hp.synfast(cls, nside=nsd, pol=True, new=True))
lenmap = len(hplus)
#np.savetxt('hplus.txt', hplus, delimiter=' ', newline='\n')
#np.savetxt('hcross.txt', hcross, delimiter=' ', newline='\n')


############## CREATE HPMAP OF QUATERNIONS #######################
#make the map of 0s

#row = np.array([0.]*4)
#quatmap = [row for _ in range(lenmap)]

#### they were here

#print q2abg.quat_2_abg(q)
#write the initial quatmap
# quatmap = Q.radecpa2quat(np.rad2deg(ra_quatmap), np.rad2deg(dec_quatmap-np.pi*0.5), 0.*np.ones_like(ra_quatmap))
# np.savetxt('quatmap.txt', quatmap, delimiter=' ', newline='\n')
# quatmap_rotated = np.ones_like(quatmap)
# radecpamap_ini = Q.quat2radecpa(quatmap)
# radecpamapnumpy_ini = np.array(radecpamap_ini).T
# np.savetxt('radecpamap_ini.txt', radecpamapnumpy_ini, delimiter=' ', newline='\n')
#
# i = 0
# while i < lenmap:
#     quatmap_rotated[i] = qr.quat_mult(n,quatmap[i])
#     i+=1
#
# np.savetxt('quatmap_rotated.txt', quatmap_rotated, delimiter=' ', newline='\n') #quatmap[i] is the quaternion corresponding to initial pixel i

#rot_ip = Q.quat2pix(quatmap_rotated,nsd)[0]
#map_rot = np.zeros_like(map)
#map_rot = map[rot_ip]
#hp.mollview(map_rot)
#plt.savefig('map_rot.pdf')

theta, phi = hp.pix2ang(nsd,np.arange(lenmap)) 
fplush_map = ofs.FplusH(theta,phi)



vec_north = hp.ang2vec( 0.,0.)
vec_m = np.array(hp.pix2vec(nsd,np.arange(lenmap)))
#print vec_m
#print vec_m.shape
#print vec_north
coseta = vec_m[2,:]


# radecpamap = Q.quat2radecpa(quatmap_rotated)
# radecpamapnumpy = np.array(radecpamap).T
# np.savetxt('radecpamap.txt', radecpamapnumpy, delimiter=' ', newline='\n')
#
# quatmap_rot_pix = Q.quat2pix(quatmap_rotated,nsd)[0] #rotated pixel list
# quatmap_rot_pol = Q.quat2pix(quatmap_rotated,nsd)[1] #rotated polarization list

xH=ofs.m(0.,ofs.beta*0.5)



####### RANDOM REAL I Q U V s ######################
Istoke = np.vstack(hp.synfast(cls, nside=nsd, pol=True, new=True))
Qstoke = np.vstack(hp.synfast(cls, nside=nsd, pol=True, new=True))
Ustoke = np.vstack(hp.synfast(cls, nside=nsd, pol=True, new=True))
Vstoke = np.vstack(hp.synfast(cls, nside=nsd, pol=True, new=True))

map = hp.synfast(cls, nside=nsd, pol=True, new=True)
hp.mollview(map)
plt.savefig('Istoke.pdf')

Area_pix = 1.#what's this, recapially?



######## CHECK ALL ANGLE CONVENTIONS AND MAKE SURE THEY ALL AGREE! ##########


def rotation_pix(m_array,n): #rotates string of pixels m around QUATERNION n
    dec_quatmap,ra_quatmap = hp.pix2ang(nsd,m_array) #
    quatmap = Q.radecpa2quat(np.rad2deg(ra_quatmap), np.rad2deg(dec_quatmap-np.pi*0.5), 0.*np.ones_like(ra_quatmap)) #but maybe orientation here is actually the orientation of detector a, b? in which case, one could input it as a variable!
    quatmap_rotated = np.ones_like(quatmap)
    i = 0
    while i < lenmap:
        quatmap_rotated[i] = qr.quat_mult(n,quatmap[i])
        i+=1
    quatmap_rot_pix = Q.quat2pix(quatmap_rotated,nsd)[0] #rotated pixel list (pols are in [1])
    return quatmap_rot_pix

#def m_prime(pixm,pixn,pixn_p):  #returns pixel number of m-n+n_p
    return ofs.vect_sum_pix(ofs.vect_diff_pix(pixm,pixn,nsd),pixn_p,nsd)

def gammaI(pixm,pixn,pixm_p,pixn_p): #returns the overlap value given 3 pixels  
    m0, m1 = hp.pix2ang(nsd,pixm)
    n0, n1 = hp.pix2ang(nsd,pixn)
    n_p0, n_p1 = hp.pix2ang(nsd,pixn_p)
    m_p0, m_p1 = hp.pix2ang(nsd,pixm_p)
    return ofs.FplusH(m0,m1)*ofs.FplusL(m_p0,m_p1)+ofs.FcrossH(m0,m1)*ofs.FcrossL(m_p0,m_p1)

def gammaQ(pixm,pixn,pixm_p,pixn_p): #redo
    m = hp.pix2ang(nsd,pixm)
    n = hp.pix2ang(nsd,pixn)
    n_p = hp.pix2ang(nsd,pixn_p)
    m_p = hp.pix2ang(nsd,pixm_p)
    return ofs.FplusH(m[0],m[1])*ofs.FplusL(m_p[0],m_p[1])-ofs.FcrossH(m[0],m[1])*ofs.FcrossL(m_p[0],m_p[1])#redo

def gammaU(pixm,pixn,pixm_p,pixn_p): 
    m = hp.pix2ang(nsd,pixm)
    n = hp.pix2ang(nsd,pixn)
    n_p = hp.pix2ang(nsd,pixn_p)
    m_p = hp.pix2ang(nsd,pixm_p)
    return ofs.FplusH(m[0],m[1])*ofs.FcrossL(m_p[0],m_p[1])+ofs.FcrossH(m[0],m[1])*ofs.FplusL(m_p[0],m_p[1])

def gammaV(pixm,pixn,pixm_p,pixn_p):
    m = hp.pix2ang(nsd,pixm)
    n = hp.pix2ang(nsd,pixn)
    n_p = hp.pix2ang(nsd,pixn_p)
    m_p = hp.pix2ang(nsd,pixm_p)
    return ofs.FcrossH(m[0],m[1])*ofs.FplusL(m_p[0],m_p[1])-ofs.FplusH(m[0],m[1])*ofs.FcrossL(m_p[0],m_p[1])

def alpha(pixm,n_diff_vect,f):
    m = hp.pix2ang(nsd,pixm)
    m_vect = ofs.m(m[0]-np.pi*0.5, m[1]) #to fit *my* convention
    return 2.*np.pi*f*R_earth*np.dot(m_vect,n_diff_vect)

def HaHb_corr(n,n_p,f,nsd): #input: 2 quats
    
    pixn = Q.quat2pix(n,nsd)[0]    #get the pix from the quat
    pixn_p = Q.quat2pix(n_p,nsd)[0]
    m_array = np.arange(lenmap) #create string of pixels
    rot_m_array = rotation_pix(m_array,n)
    m_p_array = rotation_pix(rot_m_array,qr.quat_inv(n_p))
    
    print "+++ pixn, pixn_p +++"
    print pixn, pixn_p
    print "++++++"
    
    n_minus_n_p = hp.pix2ang(nsd,ofs.vect_diff_pix(pixn,pixn_p,nsd))
    n_diff_vect = ofs.m(n_minus_n_p[0]-np.pi*0.5, n_minus_n_p[1])
    
    Istoke_rot = Istoke[rot_m_array]    #remember: check interpolation
    Qstoke_rot = Qstoke[rot_m_array]
    Vstoke_rot = Vstoke[rot_m_array]
    Ustoke_rot = Ustoke[rot_m_array]
    
    #real part
    a = np.zeros_like(Istoke)
    b = np.zeros_like(Istoke)
    
    #imaginary part
    c = np.zeros_like(Istoke)
    d = np.zeros_like(Istoke)
    
    i = 0
    while i < lenmap:
        if m_array[i]<pixn or m_array[i]>pixn:
            a[i] = (Istoke_rot[i]*gammaI(m_array[i],pixn,m_p_array[i],pixn_p)+
            Qstoke_rot[i]*gammaQ(m_array[i],pixn,m_p_array[i],pixn_p)+
            Ustoke_rot[i]*gammaU(m_array[i],pixn,m_p_array[i],pixn_p)
            )*np.cos(alpha(m_array[i],n_diff_vect,f))
            b[i] = -(Vstoke_rot[i]*gammaV(m_array[i],pixn,m_p_array[i],pixn_p))*np.sin(alpha(m_array[i],n_diff_vect,f))
            c[i] = (Istoke_rot[i]*gammaI(m_array[i],pixn,m_p_array[i],pixn_p)+
            Qstoke_rot[i]*gammaQ(m_array[i],pixn,m_p_array[i],pixn_p)+
            Ustoke_rot[i]*gammaU(m_array[i],pixn,m_p_array[i],pixn_p)
            )*np.sin(alpha(m_array[i],n_diff_vect,f))
            d[i] = (Vstoke_rot[i]*gammaV(m_array[i],pixn,m_p_array[i],pixn_p))*np.cos(alpha(m_array[i],n_diff_vect,f))
        else: 
            a[i] = 0.
            b[i] = 0.
            c[i] = 0.
            d[i] = 0.
        i+=1 
    Real_part =Area_pix*0.5*np.sum(np.add(a,b)) 
    Imag_part =Area_pix*0.5*np.sum(np.add(c,d))
    return Real_part, Imag_part, pixn
    
