import numpy as np
from numpy import cos,sin
import matplotlib
import matplotlib.pyplot as plt
import healpy as hp
import math
import sys
import matplotlib.image as mpimg


#POLARIZATION TENSORS

def etheta(theta,phi,i):
    if i==0:
        return sin(theta)*cos(phi)
    elif i==1:
        return sin(theta)*sin(phi)
    elif i==2:
        return cos(theta)
    else:
        print 'error'
        return;

def ephi(theta,phi,i):
    if i==0:
        return -sin(phi)
    elif i==1:
        return cos(phi)
    elif i==2:
        return 0
    else:
        print 'error'
        return;

def eplus(theta,phi,i,j):
    return etheta(theta,phi,i)*etheta(theta,phi,j)-ephi(theta,phi,i)*ephi(theta,phi,j)

def ecross(theta,phi,i,j):
    return etheta(theta,phi,i)*ephi(theta,phi,j)+ephi(theta,phi,i)*etheta(theta,phi,j)


#LIGO DETECOR TENSORS; H:Hanford, L:Livingston

    #const.s
# sigmaH=0.
# sigmaL=0.
# beta=np.pi

R_earth=6378137  #radius of the earth in metres

##real ones:
sigmaH = (45.3 + 62.2)*np.pi/180.
sigmaL = (45.3 - 62.2)*np.pi/180.
beta = 27.2*np.pi/180.

    #arm vectors
uH=[cos(sigmaH)*sin(beta/2.),sin(sigmaH),-cos(sigmaH)*cos(beta/2.)]
vH=[-sin(sigmaH)*sin(beta/2.),cos(sigmaH),sin(sigmaH)*cos(beta/2.)]
uL=[-cos(sigmaL)*sin(beta/2.),sin(sigmaL),-cos(sigmaL)*cos(beta/2.)]
vL=[sin(sigmaL)*sin(beta/2.),cos(sigmaL),sin(sigmaL)*cos(beta/2.)]

    #detector tensors
dH=0.5*(np.outer(uH,uH)-np.outer(vH,vH))
dL=0.5*(np.outer(uL,uL)-np.outer(vL,vL))

#OVERLAP FUNCTIONS

    #beam pattern functions
def FplusH(theta,phi):            #can also make it a func of "a": detector tensor
    res=0
    i=0
    while i<3:
        j=0
        while j<3:
            res=res+dH[i,j]*eplus(theta,phi,i,j)
            j=j+1
        i=i+1
    return res

def FplusL(theta,phi):
    res=0
    i=0
    while i<3:
        j=0
        while j<3:
            res=res+dL[i,j]*eplus(theta,phi,i,j)
            j=j+1
        i=i+1
    return res

def FcrossH(theta,phi):            #can also make it a func of "a": detector tensor
    res=0
    i=0
    while i<3:
        j=0
        while j<3:
            res=res+dH[i,j]*ecross(theta,phi,i,j)
            j=j+1
        i=i+1
    return res


def FcrossL(theta,phi):
    res=0
    i=0
    while i<3:
        j=0
        while j<3:
            res=res+dL[i,j]*ecross(theta,phi,i,j)
            j=j+1
        i=i+1
    return res

    #overlap functions
def gammaIHL(theta,phi):
    return FplusH(theta,phi)*FplusL(theta,phi)+FcrossH(theta,phi)*FcrossL(theta,phi)

def gammaVHL(theta,phi):        #module of imaginary part
    return FplusH(theta,phi)*FcrossL(theta,phi)-FcrossH(theta,phi)*FplusL(theta,phi)

def gammaQHL(theta,phi):
    return FplusH(theta,phi)*FplusL(theta,phi)-FcrossH(theta,phi)*FcrossL(theta,phi)

def gammaUHL(theta,phi):            
    return FplusH(theta,phi)*FcrossL(theta,phi)+FcrossH(theta,phi)*FplusL(theta,phi)


#########################

def m(theta, phi):  #unit vector
    return [cos(phi)*sin(theta),sin(phi)*sin(theta),cos(theta)] #defined in range: 0<th<pi
    
# def vect_diff_pix(pixm,pixn,nsd): #you give it 2 pixs, it returns the pixel of their difference (psi ang info is thus lost - but may be recovered with some work)
#     decn, ran = hp.pix2ang(nsd,pixn)   #these are in hp conventions: decn goes from 0 to pi
#     decm, ram = hp.pix2ang(nsd,pixm)
#     n_vect = m(decn-np.pi*0.5, ran) #to fit *my* convention
#     m_vect = m(decm-np.pi*0.5, ram)
#     m_minus_n = np.array(m_vect)-np.array(n_vect)
#     norm = np.sqrt(np.dot(m_minus_n,m_minus_n))
#     sina = m_minus_n[2]/norm  #a is m-n dec, b is m-n ra
#     cosa = np.sqrt(1-sina**2)  #as a is in the range -pi/2:+pi/2
#     cosb = m_minus_n[0]/(norm*cosa)
#     sinb = m_minus_n[1]/(norm*cosa)
#     a = np.arctan2(sina,cosa)
#     b = np.arctan2(sinb,cosb)
#     pix_m_minus_n = hp.ang2pix(nsd,a+np.pi*0.5,b) #
#     return pix_m_minus_n
    
# def vect_sum_pix(pixn,pixm,nsd): #you give it 2 pixs, it returns the pixel of their difference (psi ang info is thus lost - but may be recovered with some work)
#     decn, ran = hp.pix2ang(nsd,pixn)   #these are in hp conventions: decn goes from 0 to pi
#     decm, ram = hp.pix2ang(nsd,pixm)
#     n_vect = m(decn-np.pi*0.5, ran) #to fit *my* convention
#     m_vect = m(decm-np.pi*0.5, ram)
#     m_plus_n = np.array(m_vect)+np.array(n_vect)
#     norm = np.sqrt(m_plus_n.dot(m_plus_n))
#     sina = m_plus_n[2]/norm  #a is m-n dec, b is m-n ra
#     cosa = np.sqrt(1-sina**2)  #as a is in the range -pi/2:+pi/2
#     cosb = m_plus_n[0]/(norm*cosa)
#     sinb = m_plus_n[1]/(norm*cosa)
#     a = np.arctan2(sina,cosa)
#     b = np.arctan2(sinb,cosb)
#     pix_m_plus_n = hp.ang2pix(nsd,a+np.pi*0.5,b)  #
#     return pix_m_plus_n
#
