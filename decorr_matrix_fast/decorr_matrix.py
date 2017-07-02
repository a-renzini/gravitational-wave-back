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
import matplotlib.pyplot as plt
import stokes_map as sm
import imageio

#set the frequency:

f = 1

#set the maps:

nsd = 16
Q = qp.QMap(nside=nsd, pol=True, accuracy='low',
            fast_math=True, mean_aber=True)

lenmap =  hp.nside2npix(nsd)
theta, phi = hp.pix2ang(nsd,np.arange(lenmap)) 
m_array = np.arange(lenmap)
        
#Initial gammas, in the 'great circle' frame
gammaI = ofs.gammaIHL(theta,phi)
gammaQ = ofs.gammaQHL(theta,phi)
gammaU = ofs.gammaUHL(theta,phi)
gammaV = ofs.gammaVHL(theta,phi)

# hp.mollview(gammaI)
# plt.savefig('gammaI.pdf')         ##('hanning%s.pdf' % num)

#Rotation of the gammas around a n quat

n = sm.n_n_p_arrays()[0]
n_p = sm.n_n_p_arrays()[1]

#build alpha functs
cos_alpha = [np.zeros(lenmap)]*len(n)
sin_alpha = [np.zeros(lenmap)]*len(n)

i = 0
j = 0

while i<len(n):
    pixn = Q.quat2pix(n[i],nsd)[0]   
    pixn_p = Q.quat2pix(n_p[i],nsd)[0]
    while j<lenmap:
        n_minus_n_p = hp.pix2ang(nsd,ofs.vect_diff_pix(pixn,pixn_p,nsd))
        n_diff_vect = ofs.m(n_minus_n_p[0]-np.pi*0.5, n_minus_n_p[1])    
        cos_alpha[i][j] = np.cos(alpha(m_array[j],n_diff_vect,f))
        sin_alpha[i][j] = np.sin(alpha(m_array[j],n_diff_vect,f))
        j+=1
    i+=1
hp.mollview(cos_alpha[1])
plt.savefig('trials/cos_alpha.pdf')
    
## GIF MAKER

s = raw_input('Would you like to make a .gif of the overlap functions? (Y/n) ')
if s == 'Y':
    images = []
    i = 0
    while i<10: 
        rot_m_array = rotation_pix(m_array,n[i])    #this is the rotated array; now one simply does

        gammaI_rot = gammaI[rot_m_array]    #remember: check interpolation
    #    gammaQ_rot = gammaQ[rot_m_array]
    #    gammaU_rot = gammaU[rot_m_array]
    #    gammaV_rot = gammaV[rot_m_array]

        hp.mollview(gammaI_rot)
        plt.savefig('gammaI/gammaI%s.png' % i)
        images.append(imageio.imread('gammaI/gammaI%s.png' % i))
        i+=1
    imageio.mimsave('gammaI.gif', images)
    
###

## ROTATING THE GAMMAR FUNCTIONS: Elements of the DECORR matrix & of in the data vector

i = 0
while i<len(n): 
    rot_m_array = rotation_pix(m_array,n[i])    #this is the rotated array; let's pretend counter-rotation is just inverting the order of the elements !!TO BE CHECKED!! . Now one simply does

    gammaRI_rot = gammaI[rot_m_array]*cos_alpha[i][rot_m_array]
    gammaRQ_rot = gammaQ[rot_m_array]*cos_alpha[i][rot_m_array]
    gammaRU_rot = gammaU[rot_m_array]*cos_alpha[i][rot_m_array]
    gammaRV_rot = gammaV[rot_m_array]*sin_alpha[i][rot_m_array]
    
    i+=1






### BUILDING THE DECORRELATION MATRIX ####

#np.matrix([[1, gammas], [],[],[],[]])