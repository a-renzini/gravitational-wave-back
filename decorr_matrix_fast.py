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
import pickle

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

n = sm.n_n_p_arrays()[0]
n_p = sm.n_n_p_arrays()[1]

k = 0
pixn = np.arange(len(n))
pixn_p = np.arange(len(n))
while k<len(n):
    pixn[k] = Q.quat2pix(n[k],nsd)[0]   
    pixn_p[k] = Q.quat2pix(n_p[k],nsd)[0]
    k+=1

#print pixn, pixn_p

#2 build alpha functs
cos_alpha = [np.zeros(lenmap)]*len(n)
sin_alpha = [np.zeros(lenmap)]*len(n)

i = 0
notimes = 0
npix = 0
gammaRI_rtot = np.zeros_like(gammaI)
gammaRQ_rtot = np.zeros_like(gammaI)
gammaRU_rtot = np.zeros_like(gammaI)
gammaRV_rtot = np.zeros_like(gammaI)
refpix = []
refpix.append(pixn[0])
print npix

decorr_Ms = len(n)*[[[np.zeros_like(gammaI)],[np.zeros_like(gammaI)],[np.zeros_like(gammaI)],[np.zeros_like(gammaI)],[np.zeros_like(gammaI)]],[[np.zeros_like(gammaI)],[np.zeros_like(gammaI)],[np.zeros_like(gammaI)],[np.zeros_like(gammaI)],[np.zeros_like(gammaI)]],[[np.zeros_like(gammaI)],[np.zeros_like(gammaI)],[np.zeros_like(gammaI)],[np.zeros_like(gammaI)],[np.zeros_like(gammaI)]],[[np.zeros_like(gammaI)],[np.zeros_like(gammaI)],[np.zeros_like(gammaI)],[np.zeros_like(gammaI)],[np.zeros_like(gammaI)]],[[np.zeros_like(gammaI)],[np.zeros_like(gammaI)],[np.zeros_like(gammaI)],[np.zeros_like(gammaI)],[np.zeros_like(gammaI)]]]

#exit()

while i<len(n):

#the alpha functs: (depends only on pix numbers)
    j = 0
    while j<lenmap:
        n_minus_n_p = hp.pix2ang(nsd,ofs.vect_diff_pix(pixn[i],pixn_p[i],nsd))
        n_diff_vect = ofs.m(n_minus_n_p[0]-np.pi*0.5, n_minus_n_p[1])
        cos_alpha[i][j] = np.cos(alpha(m_array[j],n_diff_vect,f))
        sin_alpha[i][j] = np.sin(alpha(m_array[j],n_diff_vect,f))
        j+=1

    rot_m_array = rotation_pix(m_array,n[i])    #this is the rotated array; let's pretend counter-rotation is just inverting the order of the elements !!TO BE CHECKED!! . Now one simply does

## ROTATING THE GAMMAR FUNCTIONS: Elements of the DECORR matrix & of in the data vector

    gammaRI_rot = gammaI[rot_m_array]*cos_alpha[i][rot_m_array]
    gammaRQ_rot = gammaQ[rot_m_array]*cos_alpha[i][rot_m_array]
    gammaRU_rot = gammaU[rot_m_array]*cos_alpha[i][rot_m_array]
    gammaRV_rot = gammaV[rot_m_array]*sin_alpha[i][rot_m_array]

    if pixn[i] == refpix[npix]:
        gammaRI_rtot +=  gammaRI_rot #summing within the same pix   
        gammaRQ_rtot +=  gammaRQ_rot
        gammaRU_rtot +=  gammaRU_rot
        gammaRV_rtot +=  gammaRV_rot
        
        notimes+=1
        
    else:
        print notimes
        
        gammaRI_rtot = gammaRI_rtot/notimes     #average over pixels
        gammaRQ_rtot = gammaRQ_rtot/notimes 
        gammaRU_rtot = gammaRU_rtot/notimes 
        gammaRV_rtot = gammaRV_rtot/notimes 
        
        
        
        npix+=1
        refpix.append(pixn[i])
        notimes = 1
        
        print npix
        
        hp.mollview(gammaRI_rtot)
        plt.savefig('gammaRI_rotated/gammaRI_rotated%s.pdf' % i)
        
        decorr_Ms[npix] = [[1,gammaRI_rtot,gammaRV_rtot,gammaRU_rtot,gammaRQ_rtot],[gammaRI_rtot,gammaRI_rtot*gammaRI_rtot,gammaRI_rtot*gammaRV_rtot,gammaRI_rtot*gammaRU_rtot,gammaRI_rtot*gammaRQ_rtot],[gammaRV_rtot,gammaRV_rtot*gammaRI_rtot,gammaRV_rtot*gammaRV_rtot,gammaRV_rtot*gammaRU_rtot,gammaRV_rtot*gammaRQ_rtot],[gammaRU_rtot,gammaRU_rtot*gammaRI_rtot,gammaRU_rtot*gammaRV_rtot,gammaRU_rtot*gammaRU_rtot,gammaRU_rtot*gammaRQ_rtot],[gammaRQ_rtot,gammaRQ_rtot*gammaRI_rtot,gammaRQ_rtot*gammaRV_rtot,gammaRQ_rtot*gammaRU_rtot,gammaRQ_rtot*gammaRQ_rtot]]
                
        #with open('decorr_Ms/decorr_M%s.txt' % npix, 'wb') as fp:
        #    pickle.dump(decorr_Ms[npix], fp)
        
        #print decorr_Ms[npix]
        
        gammaRI_rtot = gammaRI_rot
        gammaRQ_rtot = gammaRQ_rot
        gammaRU_rtot = gammaRU_rot
        gammaRV_rtot = gammaRV_rot
    
    ### BUILDING THE DECORRELATION MATRIX ####

        
    i+=1

print notimes
gammaRI_rtot = gammaRI_rtot/notimes   
hp.mollview(gammaRI_rtot)
plt.savefig('gammaRI_rotated/gammaRI_rotated%s.pdf' % i)

print refpix
