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
import imageio
import pickle

#set the frequency:

f = 1   #Hz

#set the maps:

nsd = 8
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
gammaRI_tot = 0.
gammaRQ_tot = 0.
gammaRU_tot = 0.
gammaRV_tot = 0.
gammaRII_tot = 0.
gammaRIQ_tot = 0.
gammaRIU_tot = 0.
gammaRIV_tot = 0.
gammaRQQ_tot = 0.
gammaRQU_tot = 0.
gammaRQV_tot = 0.
gammaRUU_tot = 0.
gammaRUV_tot = 0.
gammaRVV_tot = 0.

refpix = []
refpix.append(pixn[0])
print npix

decorr_Ms = []#len(n)*[[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]]


while i<len(n):

#the alpha functs: (depends only on pix numbers)
    j = 0
    while j<lenmap:             #nsd = 16 ##
        n_minus_n_p = hp.pix2ang(16,ofs.vect_diff_pix(pixn[i],pixn_p[i],nsd))
        n_diff_vect = ofs.m(n_minus_n_p[0]-np.pi*0.5, n_minus_n_p[1])
        cos_alpha[i][j] = np.cos(alpha(m_array[j],n_diff_vect,f))
        sin_alpha[i][j] = np.sin(alpha(m_array[j],n_diff_vect,f))
        j+=1

#    rot_m_array = rotation_pix(m_array,quat_inv(n[i]))    #this is the rotated array; let's pretend counter-rotation is just rotating by the inverse quat !!TO BE CHECKED!! . Now one simply does

    gammaRI = np.sum(gammaI*cos_alpha[i])
    gammaRQ = np.sum(gammaQ*cos_alpha[i])
    gammaRU = np.sum(gammaU*cos_alpha[i])
    gammaRV = np.sum(gammaV*sin_alpha[i])
    
    gammaRII = np.sum(gammaI*cos_alpha[i]*gammaI*cos_alpha[i])
    gammaRIQ = np.sum(gammaQ*cos_alpha[i]*gammaI*cos_alpha[i])  # = gammaRQI
    gammaRIU = np.sum(gammaU*cos_alpha[i]*gammaI*cos_alpha[i])  # = gammaRUI
    gammaRIV = np.sum(gammaV*sin_alpha[i]*gammaI*cos_alpha[i])  # = gammaRVI
    
    gammaRQQ = np.sum(gammaQ*cos_alpha[i]*gammaQ*cos_alpha[i])
    gammaRQU = np.sum(gammaU*cos_alpha[i]*gammaQ*cos_alpha[i])  # = gammaRUQ
    gammaRQV = np.sum(gammaV*sin_alpha[i]*gammaQ*cos_alpha[i])  # = gammaRVQ
    
    gammaRUU = np.sum(gammaU*cos_alpha[i]*gammaU*cos_alpha[i])
    gammaRUV = np.sum(gammaV*sin_alpha[i]*gammaU*cos_alpha[i])  # = gammaRUV
    
    gammaRVV = np.sum(gammaV*sin_alpha[i]*gammaV*sin_alpha[i])
    
    
    if pixn[i] == refpix[npix]:

        gammaRI_tot +=  gammaRI #summing within the same pix   
        gammaRQ_tot +=  gammaRQ
        gammaRU_tot +=  gammaRU
        gammaRV_tot +=  gammaRV

        gammaRII_tot +=  gammaRII
        gammaRIQ_tot +=  gammaRIQ
        gammaRIU_tot +=  gammaRIU
        gammaRIV_tot +=  gammaRIV

        gammaRQQ_tot +=  gammaRQQ
        gammaRQU_tot +=  gammaRQU
        gammaRQV_tot +=  gammaRQV
        
        gammaRUU_tot +=  gammaRUU
        gammaRUV_tot +=  gammaRUV
        
        gammaRVV_tot +=  gammaRVV               
        
        notimes+=1
        
    else:
        print notimes
        
        gammaRI_tot = gammaRI_tot/notimes     #average over pixels
        gammaRQ_tot = gammaRQ_tot/notimes 
        gammaRU_tot = gammaRU_tot/notimes 
        gammaRV_tot = gammaRV_tot/notimes 

        gammaRII_tot = gammaRII_tot/notimes    
        gammaRIQ_tot = gammaRIQ_tot/notimes 
        gammaRIU_tot = gammaRIU_tot/notimes 
        gammaRIV_tot = gammaRIV_tot/notimes 
        
        gammaRQQ_tot = gammaRQQ_tot/notimes 
        gammaRQU_tot = gammaRQU_tot/notimes 
        gammaRQV_tot = gammaRQV_tot/notimes
        
        gammaRUU_tot = gammaRUU_tot/notimes 
        gammaRUV_tot = gammaRUV_tot/notimes

        gammaRVV_tot = gammaRVV_tot/notimes
        
        print gammaRI_tot
        
        npix+=1
        refpix.append(pixn[i])
        notimes = 1
        
        print npix
        
#        hp.mollview(gammaRI_tot,min=-0.35,max=0.35)
#        plt.savefig('GAMMA_RI/nside_8/gammaRI%s.pdf' % i)
        
        #Area_pix =
        
        decorr_Ms.append([[1.,gammaRI_tot,-gammaRV_tot,gammaRU_tot,gammaRQ_tot],[gammaRI_tot,gammaRII_tot,-gammaRIV_tot,gammaRIU_tot,gammaRIQ_tot],[-gammaRV_tot,-gammaRIV_tot,gammaRVV_tot,-gammaRUV_tot,-gammaRQV_tot],[gammaRU_tot,gammaRIU_tot,-gammaRUV_tot,gammaRUU_tot,gammaRQU_tot],[gammaRQ_tot,gammaRIQ_tot,-gammaRQV_tot,gammaRQU_tot,gammaRQQ_tot]])

        
        gammaRI_tot = gammaRI
        gammaRQ_tot = gammaRQ
        gammaRU_tot = gammaRU
        gammaRV_tot = gammaRV
        
        gammaRII_tot =  gammaRII
        gammaRIQ_tot =  gammaRIQ
        gammaRIU_tot =  gammaRIU
        gammaRIV_tot =  gammaRIV

        gammaRQQ_tot =  gammaRQQ
        gammaRQU_tot =  gammaRQU
        gammaRQV_tot =  gammaRQV
        
        gammaRUU_tot =  gammaRUU
        gammaRUV_tot =  gammaRUV
        
        gammaRVV_tot =  gammaRVV
    
    ### BUILDING THE DECORRELATION MATRIX ####

        
    i+=1

print notimes
print len(decorr_Ms)

gammaRI_tot = gammaRI_tot/notimes   
gammaRQ_tot = gammaRQ_tot/notimes 
gammaRU_tot = gammaRU_tot/notimes 
gammaRV_tot = gammaRV_tot/notimes 

gammaRII_tot = gammaRII_tot/notimes    
gammaRIQ_tot = gammaRIQ_tot/notimes 
gammaRIU_tot = gammaRIU_tot/notimes 
gammaRIV_tot = gammaRIV_tot/notimes 

gammaRQQ_tot = gammaRQQ_tot/notimes 
gammaRQU_tot = gammaRQU_tot/notimes 
gammaRQV_tot = gammaRQV_tot/notimes

gammaRUU_tot = gammaRUU_tot/notimes 
gammaRUV_tot = gammaRUV_tot/notimes

gammaRVV_tot = gammaRVV_tot/notimes

####
#test take a decorr_M at some fixed pix m (= 49 in this case), pix n (= 3, in this case) and calculate the eigens to see what happens
####

print decorr_Ms[1]

print np.linalg.eigvals(decorr_Ms[1])


#print refpix
