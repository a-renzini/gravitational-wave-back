import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.special import spherical_jn, sph_harm    
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz, hanning
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py
import datetime as dt
import pytz
import pylab
import qpoint as qp
import healpy as hp
from camb.bispectrum import threej
import quat_rotation as qr
from scipy.optimize import curve_fit
import OverlapFunctsSrc as ofs
from numpy import cos,sin
from matplotlib import cm
from scipy.linalg import block_diag

# LIGO-specific readligo.py 
import readligo as rl
import ligo_filter as lf
from gwpy.time import tconvert
from glue.segments import *

'''

LIGO ANALYSIS ROUTINES

    *Fixed Setup Constants
    *Basic Tools
    *Data Segmenter
    *Projector
    *Scanner

'''

def map_in_gauss(nside_in, noise_lvl):
    
    nside = nside_in
    
    if noise_lvl == 1: beta = 1.
    elif noise_lvl == 2: beta = 1.e-42
    elif noise_lvl == 3: beta = 1.e-43
    elif noise_lvl == 4: beta = 2.e-44

    
    lmax = nside/4
    alm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=np.complex)
    
    cls = hp.sphtfunc.alm2cl(alm)

    cls=[1]*lmax
    i=1
    while i<lmax+1:
        cls[i-1]=1./i/(i+1.)*2*np.pi
        i+=1
    
    #does synfast include the monopole?
    #realistic case: monopole should be larger then others, then dipole 1.e-2
    
    return (np.vstack(hp.synfast(cls, nside=nside, pol=True, new=True, verbose = False)).flatten())*beta

class Generator(object):
    
    def __init__(self,nside, sig_name):     #use nside_in! 
        
        self.nside = nside
        self.lmax = self.nside/2
        self.a_lm = np.zeros(hp.Alm.getidx(self.lmax,self.lmax,self.lmax)+1,dtype=complex)
        
        if sig_name == 'mono':    

            #self.a_lm = np.zeros(hp.Alm.getidx(self.lmax,self.lmax,self.lmax)+1,dtype=complex)

            #self.a_lm[4] = 1.
            self.a_lm[0] = 1.
            #cls = hp.sphtfunc.alm2cl(a_lm)

            # cls=[1]*nside
            # i=0
            # while i<nside:
            #     cls[i]=1./(i+1.)**2.
            #     i+=1
        
        elif sig_name == '2pol1':

            #self.a_lm[4] = 1.
            self.a_lm[1] = 1.
        
        elif sig_name == '2pol2':
            l = 1
            m = 0
            idx = hp.Alm.getidx(self.lmax,l,abs(m))
            self.a_lm[idx] = 1.
            l = 1
            m = 1
            #print self.lmax,l,abs(m)
            idx = hp.Alm.getidx(self.lmax,l,abs(m))
            #print idx
            self.a_lm[idx] = 1.  
                      
        elif sig_name == '4pol1':

            l = 2
            m = 0
            idx = hp.Alm.getidx(self.lmax,l,abs(m))
            self.a_lm[idx] = 1.

            
        elif sig_name == '4pol2':

            l = 2
            m = 1
            idx = hp.Alm.getidx(self.lmax,l,abs(m)) 
            self.a_lm[idx] = 1.
            l = 2
            m = 2
            idx = hp.Alm.getidx(self.lmax,l,abs(m))
            self.a_lm[idx] = 1. 
        
        Istoke = hp.sphtfunc.alm2map(self.a_lm, nside, verbose= False)
            
            
    def get_a_lm(self):
        return self.a_lm

class Dect(object):
    
    def __init__(self,nside, dect_name):
        
        self.R_earth=6378137
        self.beta = 27.2*np.pi/180.
        self._nside = nside
        lmax = nside/2                                                                                                        
        self.lmax = lmax
        self.name = dect_name
        
        self.Q = qp.QPoint(accuracy='low', fast_math=True, mean_aber=True)#, num_threads=1)
        
        # Configuration: radians and metres, Earth-centered frame
        if dect_name =='H1':
            
            self._lon = -2.08405676917
            self._lat = 0.81079526383
            self._elev = 142.554
            self._vec = np.array([-2.16141492636e+06, -3.83469517889e+06, 4.60035022664e+06])            
            self._alpha = (171.8)*np.pi/180.

        
        elif dect_name =='L1':
            self._lon = -1.58430937078
            self._lat = 0.53342313506           
            self._elev = -6.574
            self._vec = np.array([-7.42760447238e+04, -5.49628371971e+06, 3.22425701744e+06])
            
            self._alpha = (243.0)*np.pi/180.

      
        elif dect_name =='V1':
            self._lon = 0.18333805213
            self._lat = 0.76151183984
            self._elev = 51.884
            self._vec = np.array([4.54637409900e+06, 8.42989697626e+05, 4.37857696241e+06])
            
            self._alpha = 116.5*np.pi/180.         #np.radians()
        
        elif dect_name =='G':
            self._lon = 0.1710
            self._lat = 0.8326
            self._vec = np.array([4.2296e+6, 730394., 4.7178e+6])
            
            self._alpha = 68.8*np.pi/180.         #np.radians()
            
        elif dect_name =='K':
            self._lon = 2.39424267
            self._lat = 0.63268185
            self._vec = np.array([-3.7728e+6,3.4961e+6,3.77145e+6])
            
            self._alpha = 225.0*np.pi/180.         #np.radians()

        #######################
        
        elif dect_name =='A':
            self._lon = 0.
            self._lat = 0.
            self._vec = np.array([self.R_earth,0.,0.])
            
            self._alpha = 0.         #np.radians()
            
        elif dect_name =='B':
            self._lon = np.pi/2.
            self._lat = 0.
            self._vec = np.array([0.,self.R_earth,0.])
            
            self._alpha = np.pi         #np.radians()
            
        elif dect_name =='C':
            self._lon = np.pi
            self._lat = 0.
            self._vec = np.array([-self.R_earth,0.,0.])
            
            self._alpha = 0.         #np.radians()
        
        elif dect_name =='D':
            self._lon = -np.pi/2.
            self._lat = 0.
            self._vec = np.array([0.,-self.R_earth,0.])
            
            self._alpha = np.pi         #np.radians()
            
        elif dect_name =='E':
            self._lon = 0.0001
            self._lat = -np.pi/2.-0.0001
            self._vec = np.array([0.,-self.R_earth*1.e-6,-self.R_earth])
            
            self._alpha = 0.         #np.radians()
            
        elif dect_name =='F':
            self._lon = 0.0001-0.0001
            self._lat = np.pi/2.
            self._vec = np.array([0.,-self.R_earth*1.e-6,self.R_earth])
            
            self._alpha = 0.         #np.radians()
        
        else:
            dect_name = __import__(dect_name)
            #import name
            self._lon = dect_name.lon
            self._lat = dect_name.lat
            self._vec = dect_name.vec
            self._alpha = dect_name.alpha
        
        
        self._ph = self._lon + 2.*np.pi;
        self._th = self._lat + np.pi/2.
        
        self._alpha = np.pi/180.
        self._u = self.u_vec()
        self._v = self.v_vec()
        
        
        self.npix = hp.nside2npix(self._nside)
        theta, phi = hp.pix2ang(self._nside,np.arange(self.npix))
        self.Fplus = self.Fplus(theta,phi)
        self.Fcross = self.Fcross(theta,phi)
        self.dott = self.dott(self._vec)
        # print 'fplus_int ', dect_name
        # print np.sum(self.Fplus)*4.*np.pi/self.npix
        # print 'fcross_int ', dect_name
        # print np.sum(self.Fcross)*4.*np.pi/self.npix
        # print 'fplusfplus_int ', dect_name
        # print np.sum(self.Fplus*self.Fplus+self.Fcross*self.Fcross)*4.*np.pi/self.npix
        # #print np.sum(self.Fcross*self.Fcross)*4.*np.pi/self.npix
        #
        # print 'Fplus[0]'
        # print hp.map2alm(self.Fplus, lmax = lmax)[0]
        #
        #hp.mollview(self.Fplus)
        #plt.savefig('Fp.pdf')
        
        
        if lmax>0:
        # cache 3j symbols
            self.threej_0 = np.zeros((2*lmax+1,2*lmax+1,2*lmax+1))
            self.threej_m = np.zeros((2*lmax+1,2*lmax+1,2*lmax+1,2*lmax+1,2*lmax+1))
            for l in range(lmax+1):
                for m in range(-l,l+1):
                    for lp in range(lmax+1):
                        lmin0 = np.abs(l - lp)
                        lmax0 = l + lp
                        self.threej_0[lmin0:lmax0+1,l,lp] = threej(l, lp, 0, 0)
                        for mp in range(-lp,lp+1):
                            # remaining m index
                            mpp = -(m+mp)
                            lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                            lmax_m = l + lp
                            self.threej_m[lmin_m:lmax_m+1,l,lp,m,mp] = threej(lp, l, mp, m) ###
        
        
    def lon(self):
        return self._lon
    def lat(self):
        return self._lat
    def th(self):
        return self._th
    def ph(self):
        return self._ph
    def elev(self):
        return self._elev
    def vec(self):
        return self._vec
    
    def u_(self):
        th = self._th
        ph = self._ph
        a = -cos(th)*cos(ph)
        b = -cos(th)*sin(ph)
        c = sin(th)
        norm = np.sqrt(a**2+b**2+c**2)
        return 1./norm * np.array([a,b,c])
        
    def v_(self):
        th = self._th
        ph = self._ph
        a = -sin(th)*sin(ph)
        b = sin(th)*cos(ph)
        c = 0.
        norm = np.sqrt(a**2+b**2+c**2)
        vec = np.array([a,b,c])
        if norm == 0.: 
            norm = 1.
        if self.name == 'E':
            vec = np.array([0.,-1.,0.])
        if self.name == 'F':
            vec = np.array([0.,1.,0.])
        return 1./norm * vec     
        
    def u_vec(self):
        a_p = self._alpha - np.pi/4.
        return self.u_()*cos(a_p) - self.v_()*sin(a_p)
        
    def v_vec(self):
        a_p = self._alpha - np.pi/4.
        return self.u_()*sin(a_p) + self.v_()*cos(a_p)
        
    def d_tens(self):
        return 0.5*(np.outer(self._u,self._u)-np.outer(self._v,self._v))   
        
    def Fplus(self,theta,phi):            
        d_t = self.d_tens()
        res=0
        i=0
        while i<3:
            j=0
            while j<3:
                res=res+d_t[i,j]*ofs.eplus(theta,phi,i,j)
                j=j+1
            i=i+1
            
        return res
        
    def Fcross(self,theta,phi): 
        
        d_t = self.d_tens()
        
        res=0
        i=0
        while i<3:
            j=0
            while j<3:
                res=res+d_t[i,j]*ofs.ecross(theta,phi,i,j)
                j=j+1
            i=i+1
        return res

    def get_Fplus_lm(self):
        return hp.map2alm(self.Fplus,self.lmax, pol=False) 
    
    def get_Fcross_lm(self):
        return hp.map2alm(self.Fcross,self.lmax, pol=False)

    def rot_Fplus_lm(self,q_x):
        rot_m_array = self.rotation_pix(np.arange(self.npix), q_x) #rotating around the bisector of the gc 
        
        Fplus_rot = self.Fplus[rot_m_array]
        return hp.map2alm(Fplus_rot,self.lmax, pol=False) 
    
    def rot_Fcross_lm(self,q_x):
        rot_m_array = self.rotation_pix(np.arange(self.npix), q_x) #rotating around the bisector of the gc 
        
        Fcross_rot = self.Fcross[rot_m_array]
        return hp.map2alm(Fcross_rot,self.lmax, pol=False)

    def dott(self,x_vect):

        m = hp.pix2ang(self._nside,np.arange(self.npix))
        m_vect = np.array(ofs.m(m[0], m[1])) #fits *my* convention: 0<th<pi, like for hp
        #print self.R_earth*np.dot(m_vect.T,x_vect)

        return np.dot(m_vect.T,x_vect)  #Rearth is in x_vect!
    
    def get_Fplus(self):
        return self.Fplus

    def get_Fcross(self):
        return self.Fcross    
        
    def get_dott(self):
        return self.dott

    def coupK(self,l,lp,lpp,m,mp):
        return np.sqrt((2*l+1.)*(2*lp+1.)*(2*lpp+1.)/4./np.pi)*self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp]

    def rotation_pix(self,m_array,n): #rotates string of pixels m around QUATERNION n
        nside = hp.npix2nside(len(m_array))
        dec_quatmap,ra_quatmap = hp.pix2ang(nside,m_array) #
        quatmap = self.Q.radecpa2quat(np.rad2deg(ra_quatmap), np.rad2deg(dec_quatmap-np.pi*0.5), 0.*np.ones_like(ra_quatmap)) #but maybe orientation here is actually the orientation of detector a, b? in which case, one could input it as a variable!
        quatmap_rotated = np.ones_like(quatmap)
        i = 0
        while i < len(m_array): 
            quatmap_rotated[i] = qr.quat_mult(n,quatmap[i])
            i+=1
        quatmap_rot_pix = self.Q.quat2pix(quatmap_rotated,nside)[0] #rotated pixel list (polarizations are in [1])
        return quatmap_rot_pix

    def simulate(self,freqs,q_x,typ = 'mono'):
        sim = []
        nside = self._nside
        gen = Generator(nside,typ)
        lmax = self.lmax
        
        pix_x = self.Q.quat2pix(q_x, nside=nside, pol=True)[0]
        th_x, ph_x = hp.pix2ang(nside,pix_x)
        
        hplm = gen.get_a_lm()
        hclm = gen.get_a_lm()
        Fplm = self.rot_Fplus_lm(q_x)
        Fclm = self.rot_Fcross_lm(q_x)
                        
        c = 3.e8
        
        if typ == 'mono':
            lminl = 0
            lmaxl = 0
            lmaxm = 0
        
        elif typ == '2pol1':
            lminl = 1
            lmaxl = 1
            lmaxm = 0
            
        elif typ == '2pol2':
            lminl = 1
            lmaxl = 1
            lmaxm = 1
        
        elif typ == '4pol1':
            lminl = 2
            lmaxl = 2
            lmaxm = 0
            
        elif typ == '4pol2':
            lminl = 2
            lmaxl = 2
            lmaxm = 2
               
        else: 
            lmaxl = lmax 
            lminl = 0
            lmaxm = 0
        sample_freqs = freqs[::500]
        sample_freqs = np.append(sample_freqs,freqs[-1])
        
        #fixed poles
        
        for f in sample_freqs:     #NEEDS TO CALL GEOMETRY METHINKS

            sim_f = 0.
            
            for l in range(lminl,lmaxl+1): #

                for m in range(-lmaxm,lmaxm+1): #
                    
                    idx_lm = hp.Alm.getidx(lmax,l,abs(m))
                    for lp in range(lmax+1): #
                        for mp in range(-lp,lp+1): #
        
                            idx_lpmp = hp.Alm.getidx(lmax,lp,abs(mp))
                            #print '(',idx_lm, idx_ltmt, ')'

                            # remaining m index
                            mpp = -(m+mp)
                            lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                            lmax_m = l + lp
                    
                            for idxl, lpp in enumerate(range(lmin_m,lmax_m+1)):
                        
                                if m>0:
                                    if mp>0:
                                        sim_f+=4*np.pi*(0.+1.j)**lpp*(spherical_jn(lpp, 2.*np.pi*(f)*self.R_earth/c)
                                        *np.conj(sph_harm(mpp, lpp, th_x, ph_x))*self.coupK(lp,l,lpp,mp,m)
                                        *(hplm[idx_lm]*Fplm[idx_lpmp]+hclm[idx_lm]*Fclm[idx_lpmp]) )

                            
                                    else:
                                        sim_f+=4*np.pi*(0.+1.j)**lpp*(spherical_jn(lpp, 2.*np.pi*(f)*self.R_earth/c)
                                        *np.conj(sph_harm(mpp, lpp, th_x, ph_x))*self.coupK(lp,l,lpp,mp,m)
                                        *(hplm[idx_lm]*np.conj(Fplm[idx_lpmp])+hclm[idx_lm]*np.conj(Fclm[idx_lpmp])) )*(-1)**mp

                                
                                else:
                                    if mp>0:
                                        sim_f+=4*np.pi*(0.+1.j)**lpp*(spherical_jn(lpp, 2.*np.pi*(f)*self.R_earth/c)
                                        *np.conj(sph_harm(mpp, lpp, th_x, ph_x))*self.coupK(lp,l,lpp,mp,m)
                                        *(np.conj(hplm[idx_lm])*Fplm[idx_lpmp]+np.conj(hclm[idx_lm])*Fclm[idx_lpmp]) )*(-1)**m

                            
                                    else:
                                        sim_f+=4*np.pi*(0.+1.j)**lpp*(spherical_jn(lpp, 2.*np.pi*(f)*self.R_earth/c)
                                        *np.conj(sph_harm(mpp, lpp, th_x, ph_x))*self.coupK(lp,l,lpp,mp,m)
                                        *(np.conj(hplm[idx_lm])*np.conj(Fplm[idx_lpmp])+np.conj(hclm[idx_lm])*np.conj(Fclm[idx_lpmp])) )*(-1)**m*(-1)**mp
            

            sim.append(sim_f)
        sim_func = interp1d(sample_freqs,sim)

        #phases = np.exp(1.j*np.random.random_sample(len(freqs))*2.*np.pi)/np.sqrt(2.)

        sim = np.array(sim_func(freqs))#*np.array(phases)

        return sim#len(freqs)*4         #for the correct normalisation



class Telescope(object):

    def __init__(self, nside_in,nside_out,fs, low_f, high_f, dects, maptyp, this_path, noise_lvl=1, alpha=3., f0=1., data_run = 'O1',npol=1): #Dect list
    
        self.Q = qp.QPoint(accuracy='low', fast_math=True, mean_aber=True)#, num_threads=1)
        
        self.pol = True #make this obsolete
        self.npol = int(npol)
            
        self.this_path = this_path
        self.data_run = data_run
        
        self.R_earth = 6378137
        self._nside_in = nside_in
        self._nside_out = nside_out
        self.fs = fs
        self.low_f = low_f
        self.high_f = high_f
        
        self.alpha = alpha
        self.f0 = f0

        # ********* Fixed Setup Constants *********

        # Configuration: radians and metres, Earth-centered frame
        
        
        #dects = ['H1','L1','V1']
        self.detectors = np.array([])
        for d in dects: 
            self.detectors = np.append(self.detectors,Dect(nside_in,d))
        
        self.ndet = len(self.detectors)
        
        ##self.H1 = Dect(nside_in,'H1')
        ##self.L1 = Dect(nside_in, 'L1')
        ##self.V1 = Dect(nside_in, 'V1')
        
        '''
        make these into lists probably:
        '''
        #for dect in listdect:
        #    self.vec2azel(dect.vec,self.L1.vec())
        
        self._nbase = int(self.ndet*(self.ndet-1)/2)
        self.combo_tuples = []
        
        
        
        for j in range(1,self.ndet):
            for k in range(j):
                self.combo_tuples.append([k,j])

        
        # work out viewing angle of baseline H1->L1
        self.az_b = np.zeros(self._nbase)
        self.el_b = np.zeros(self._nbase)
        self.baseline_length = np.zeros(self._nbase)
        
        #self.vec2azel(self.H1.vec(),self.L1.vec())
        # position of mid point and angle of great circle connecting to observatories
        self.latMid = np.zeros(self._nbase)
        self.lonMid = np.zeros(self._nbase)
        self.azMid = np.zeros(self._nbase)
        
        #boresight and baseline quaternions
        
        
        
        for i in range(self._nbase):
            a, b = self.combo_tuples[i]
            self.az_b[i], self.el_b[i], self.baseline_length[i] = self.vec2azel(self.detectors[a].vec(),self.detectors[b].vec())
            self.latMid[i], self.lonMid[i], self.azMid[i] = self.midpoint(self.detectors[a].lat(),self.detectors[a].lon(),self.detectors[b].lat(),self.detectors[b].lon())
        # gamma functs
        
        
        #np.savez('baseline_lengths.npz', baseline_length = self.baseline_length, dects = dects, combos = self.combo_tuples )
        #print 'saved baseline_lengths.npz'
        
        self.npix_in = hp.nside2npix(self._nside_in)
        self.npix_out = hp.nside2npix(self._nside_out)

        # calculate overlap functions
        # TODO: integrate this with general detector table
        #theta, phi = hp.pix2ang(self._nside,np.arange(self.npix)) 
        
        self.gammaI = []
        self.gammaV = []
        self.gammaQ = []
        self.gammaU = []
        
        self.gammaMatrix = []
        #gammas are all real ? - gammaV should have an i in front?
        
        for i in range(self._nbase):
            a, b = self.combo_tuples[i]
            
            self.gammaI.append((self.detectors[a].get_Fplus()*self.detectors[b].get_Fplus()+self.detectors[a].get_Fcross()*self.detectors[b].get_Fcross()))
            
            #self.gammaI.append((5./(8.*np.pi))*(self.detectors[a].get_Fplus()*self.detectors[b].get_Fplus()+self.detectors[a].get_Fcross()*self.detectors[b].get_Fcross()))
                        
            if self.pol == True:
            
                self.gammaV.append(1.j*(self.detectors[a].get_Fplus()*self.detectors[b].get_Fcross()-self.detectors[a].get_Fcross()*self.detectors[b].get_Fplus())) #edit : according to seto-taruya, it's 1.j; used to be a -1.! possibly due to CMB-convention?

                self.gammaQ.append((self.detectors[a].get_Fplus()*self.detectors[b].get_Fplus()-self.detectors[a].get_Fcross()*self.detectors[b].get_Fcross()))

                self.gammaU.append((self.detectors[a].get_Fplus()*self.detectors[b].get_Fcross()+self.detectors[a].get_Fcross()*self.detectors[b].get_Fplus()))
                
                #self.gammaMatrix.append(block_diag(self.gammaI[i], self.gammaV[i],self.gammaQ[i],self.gammaU[i]))
                
                
                #Simulation tools        
        
        self.noise_lvl = noise_lvl
        self.beta = 1.
        
        # plt.figure()
        # hp.mollview(self.gammaI[0])
        # plt.savefig('gammai.pdf' )
            
        # plt.figure()
        # hp.mollview(-1.j*self.gammaV[0])
        # plt.savefig('gammav.pdf' )
        
        if noise_lvl == 1: beta = 1.
        elif noise_lvl == 2: beta = 1.e-42
        elif noise_lvl == 3: beta = 1.e-43
        elif noise_lvl == 4: beta = 2.e-44
        
        
        self.beta = beta

        #print 'beta is', self.beta
        if self.npol ==1:
            input_map = np.array([self.get_map_in(maptyp)])
        
        elif self.npol ==2:
            Iscale = 2.
            Vscale = 1.
            
            Imap = Iscale * self.get_map_in(maptyp)
            Vmap = Vscale * self.get_map_in(maptyp)
            
            input_map = np.array([Imap,Vmap])#.flatten()
            
        elif self.npol == 4:
            
            # these should come from the master
            Iscale = 2.
            Vscale = 0.#1.
            Qscale = 0.#1.
            
            Imap = Iscale * self.get_map_in(maptyp)
            Vmap = Vscale * self.get_map_in(maptyp)
            Qmap = Qscale * self.get_map_in(maptyp)
            
            Umap = Qmap #np.sqrt(Imap**2-Vmap**2-Qmap**2) 
            
            ## negative maps..??
            input_map = np.array([Imap,Vmap,Qmap,Umap])#.flatten()
        
        #else: print 'npol doesnt match known pol modes'
         
        self.map_in = input_map.copy()
        
        #print self.map_in
        
        # plt.figure()
        # hp.mollview(self.map_in)
        # plt.savefig('map_in.pdf' )
        
    # ********* Maps IN *************
    
    def get_map_in(self, maptyp):
        
        nside = self._nside_in
        beta = self.beta
        
        lmax = nside/2        #or not?

        alm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=np.complex)
        
        if maptyp == None: 
            map_in = hp.alm2map(alm,nside=self._nside_in,verbose = False)
            
        elif maptyp == '1pole':
            idx = hp.Alm.getidx(lmax,0,0)
            alm[idx] = (1.+ 0.j)*beta

            map_in = hp.alm2map(alm,nside=self._nside_in,verbose = False)
        
        elif maptyp == '2pole':
            idx = hp.Alm.getidx(lmax,1,1)
            alm[idx] = (1.+ 0.j)*beta
            
            map_in = hp.alm2map(alm,nside=self._nside_in,verbose = False)

        elif maptyp == '2pole1':
            idx = hp.Alm.getidx(lmax,1,0)
            alm[idx] = (1.+ 0.j)*beta
            
            idx = hp.Alm.getidx(lmax,1,1)
            alm[idx] = (.58+ .73j)*beta
            #idx = hp.Alm.getidx(lmax,1,1)
            #alm[idx] = 1.+ 0.j
            
            map_in = hp.alm2map(alm,nside=self._nside_in,verbose = False)
        
        elif maptyp == '4pole':
            idx = hp.Alm.getidx(lmax,2,2)
            alm[idx] = (1.+ 0.j)*beta
            
            map_in = hp.alm2map(alm,nside=self._nside_in,verbose = False)
            
        elif maptyp == '8pole':
            idx = hp.Alm.getidx(lmax,3,3)
            alm[idx] = (1.+ 0.j)*beta
            
            map_in = hp.alm2map(alm,nside=self._nside_in,verbose = False)
        
        elif maptyp == '8pole1':
            idx = hp.Alm.getidx(lmax,3,3)
            alm[idx] = (1.+ .72j)*beta
            idx = hp.Alm.getidx(lmax,3,2)
            alm[idx] = (.58+ .67j)*beta
            
            map_in = hp.alm2map(alm,nside=self._nside_in,verbose = False)
        
        elif maptyp == 'gauss':
            map_file = np.load('%s/map_in%s.npz' % (self.this_path,self.noise_lvl))
            map_in = map_file['map_in']
        
        elif maptyp == 'checkfile':
            checkdata = np.load(self.this_path + '/checkfile.npz')
            map_in = checkdata['map_in']
            map_in = hp.ud_grade(map_in,nside_out = self._nside_in)
            
        # elif maptyp == 'gauss2':
        #     lmax = lmax
        #     cls = hp.sphtfunc.alm2cl(alm)
        #
        #     cls=[1]*lmax
        #     i=0
        #     while i<lmax:
        #         cls[i]=1./(i+1.)**2.
        #         i+=1
        #
        #     map_in = (np.vstack(hp.synfast(cls, nside=nside, pol=True, new=True)).flatten())*alpha
        
        elif maptyp == 'planck' or maptyp == 'planck_poi':
            fwhm = 5*np.pi/180.
            planckmap = hp.read_map('%s/COM_CompMap_dust-commander_0256_R2.00.fits' % self.this_path)
            planckmap = hp.sphtfunc.smoothing(planckmap,fwhm = fwhm, verbose = False)
            map_in = (hp.ud_grade(planckmap,nside_out = self._nside_in))
            max_in = max(map_in)
            map_in = map_in/max_in*beta
        
        return map_in
        
    # ********* Basic Tools *********
    
    def gaussian(self,x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    def halfgaussian(self,x, mu, sig):
        out = np.ones_like(x)
        out[int(mu):]= np.exp(-np.power(x[int(mu):] - mu, 2.) / (2 * np.power(sig, 2.)))
        return out    
    
    def owindow(self,l):
        x = np.linspace(0.0, l, num=l)
        gauss_lo = self.halfgaussian(x,2.*l/82.,l/82.)
        gauss_hi = self.halfgaussian(x,l-l/82.*6.,l/82.)
        win = (1.-gauss_lo)*(gauss_hi)
        
        # plt.figure()
        # plt.plot(win, color = 'r')
        # plt.savefig('win.png' )
        
        return win
    
    def ffit(self,f,c,d,e):
        return e*((c/(0.1+f))**(4.)+(f/d)**(2.)+1.)#w*(1.+(f_k*f**(-1.))**1.+(f/h_k)**b)

    def ffit2(self,f,c,d,e):
        return e*((c/(0.1+f))**(6.)+(f/d)**(2.)+1.)#w*(1.+(f_k*f**(-1.))**1.+(f/h_k)**b)        

    def rotation_pix(self,m_array,n): #rotates string of pixels m around QUATERNION n
        
        nside = hp.npix2nside(len(m_array))
        dec_quatmap,ra_quatmap = hp.pix2ang(nside,m_array) #
        
        quatmap = self.Q.radecpa2quat(np.rad2deg(ra_quatmap), np.rad2deg(dec_quatmap-np.pi*0.5), np.zeros_like(ra_quatmap)) #but maybe orientation here is actually the orientation of detector a, b? in which case, one could input it as a variable!
        quatmap_rotated = np.ones_like(quatmap)
        
        i = 0
        while i < len(m_array): 
            quatmap_rotated[i] = qr.quat_mult(n,quatmap[i]) ###
            i+=1
            
        quatmap_rot_pix, sin2psi, cos2psi = self.Q.quat2pix(quatmap_rotated,nside) #rotated pixel list (polarizations are in [1])
        
        sin4psi = 2*sin2psi*cos2psi
        cos4psi = cos2psi**2 - sin2psi**2 
        
        return quatmap_rot_pix, sin4psi, cos4psi

    def E_f(self,f,alpha=3.,f0=1.):
        #print 'alpha', alpha, 'f0', f0
        #exit()
        return (f/f0)**(alpha-3.)
    
    def coupK(self,l,lp,lpp,m,mp):
        return np.sqrt((2*l+1.)*(2*lp+1.)*(2*lpp+1.)/4./np.pi)*self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp]

    def vec2azel(self,v2,v1):   ##  edit!! changed INPUT ANGLES to suit our convention (CMB) - make sure this is what we want. WAS V1,V2
        # calculate the viewing angle from location at v1 to v2
        # Cos(elevation+90) = (x*dx + y*dy + z*dz) / Sqrt((x^2+y^2+z^2)*(dx^2+dy^2+dz^2))
        # Cos(azimuth) = (-z*x*dx - z*y*dy + (x^2+y^2)*dz) / Sqrt((x^2+y^2)(x^2+y^2+z^2)(dx^2+dy^2+dz^2))
        # Sin(azimuth) = (-y*dx + x*dy) / Sqrt((x^2+y^2)(dx^2+dy^2+dz^2))
        
        v = v2-v1      
        d = np.sqrt(np.dot(v,v))
        cos_el = np.dot(v2,v)/np.sqrt(np.dot(v2,v2)*np.dot(v,v))
        el = np.arccos(cos_el)-np.pi/2.
        cos_az = (-v2[2]*v2[0]*v[0] - v2[2]*v2[1]*v[1] + (v2[0]**2+v2[1]**2)*v[2])/np.sqrt((v2[0]**2+v2[1]**2)*np.dot(v2,v2)*np.dot(v,v))
        sin_az = (-v2[1]*v[0] + v2[0]*v[1])/np.sqrt((v2[0]**2+v2[1]**2)*np.dot(v,v))
        az = np.arctan2(sin_az,cos_az)

        return az, el, d

    def midpoint(self,lat1,lon1,lat2,lon2):
        # http://www.movable-type.co.uk/scripts/latlong.html 
        Bx = np.cos(lat2) * np.cos(lon2-lon1)
        By = np.cos(lat2) * np.sin(lon2-lon1)

        latMid = np.arctan2(np.sin(lat1) + np.sin(lat2),np.sqrt((np.cos(lat1)+Bx)*(np.cos(lat1)+Bx) + By*By))
        lonMid = lon1 + np.arctan2(By, np.cos(lat1) + Bx)

        # bearing of great circle at mid point (azimuth wrt local North) 
        y = np.sin(lon2-lonMid) * np.cos(lat2);
        x = np.cos(latMid)*np.sin(lat2) - np.sin(latMid)*np.cos(lat2)*np.cos(lon2-lonMid);
        brng = np.degrees(np.arctan2(y, x));

        return latMid,lonMid, brng

    def geometry(self,ct_split, pol = False):		#ct_split = ctime_i
        
        #returns the baseline pixel p and the boresight quaternion q_n
        nside = self._nside_in
        mid_idx = int(len(ct_split)/2)
        
        q_b = []
        q_n = []
        p = np.zeros(self._nbase, dtype = int)
        s2p = np.zeros(self._nbase)
        c2p = np.zeros(self._nbase)
        n = np.zeros_like(p)
        
        for i in range(self._nbase):
            a, b = self.combo_tuples[i]

            q_b.append(self.Q.rotate_quat(self.Q.azel2bore(np.degrees(self.az_b[i]), np.degrees(self.el_b[i]), None, None, np.degrees(self.detectors[b].lon()), np.degrees(self.detectors[b].lat()), ct_split[mid_idx])[0]))
            q_n.append(self.Q.rotate_quat(self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid[i]), np.degrees(self.latMid[i]), ct_split[mid_idx])[0]))
            p[i], s2p[i], c2p[i] = self.Q.quat2pix(q_b[i], nside=nside, pol=True)
            n[i] = self.Q.quat2pix(q_n[i], nside=nside, pol=True)[0]
            
        #p, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True)
        #n, s2n, c2n = self.Q.quat2pix(q_n, nside=nside, pol=True)  
        #theta_b, phi_b = hp.pix2ang(nside,p)
        
        if pol == False: return p, q_n, n
        else : return p, s2p, c2p, q_n, n

    def geometry_up(self,ct_split, pol = False):		#ct_split = ctime_i
        
        #returns the baseline pixel p and the boresight quaternion q_n
        nside = self._nside_in*8
        #print 'nside for bpix is: ' , nside
        mid_idx = int(len(ct_split)/2)
        
        q_b = []
        q_n = []
        p = np.zeros(self._nbase, dtype = int)
        s2p = np.zeros(self._nbase)
        c2p = np.zeros(self._nbase)
        n = np.zeros_like(p)
        
        for i in range(self._nbase):
            a, b = self.combo_tuples[i]
            q_b.append(self.Q.rotate_quat(self.Q.azel2bore(np.degrees(self.az_b[i]), np.degrees(self.el_b[i]), None, None, np.degrees(self.detectors[b].lon()), np.degrees(self.detectors[b].lat()), ct_split[mid_idx])[0]))
            q_n.append(self.Q.rotate_quat(self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid[i]), np.degrees(self.latMid[i]), ct_split[mid_idx])[0]))
            p[i], s2p[i], c2p[i] = self.Q.quat2pix(q_b[i], nside=nside, pol=True)
            n[i] = self.Q.quat2pix(q_n[i], nside=nside, pol=True)[0]
        
        #p, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True)
        #n, s2n, c2n = self.Q.quat2pix(q_n, nside=nside, pol=True)  
        #theta_b, phi_b = hp.pix2ang(nside,p)
        
        if pol == False: return p, q_n, n
        else : return p, s2p, c2p, q_n, n

    def geometry_sim(self,ct_split, pol = False):		#ct_split = ctime_i
        
        #returns the baseline pixel p and the boresight quaternion q_n
        nside = self._nside_in
        mid_idx = int(len(ct_split)/2)
        
        q_xes = []
        p = np.zeros(self._nbase, dtype = int)

        
        for i in range(self.ndet):
            q_xes.append(self.Q.rotate_quat(self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.detectors[i].lon()), np.degrees(self.detectors[i].lat()), ct_split[mid_idx])[0]))
            
            #n[i] = self.Q.quat2pix(q_n[i], nside=nside, pol=True)[0]
        
        #p, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True)
        #n, s2n, c2n = self.Q.quat2pix(q_n, nside=nside, pol=True)  
        #theta_b, phi_b = hp.pix2ang(nside,p)
        
        return q_xes
   
                
# **************** Whitening Modules *************** 

    def poissonify(self,map_in):
        norm = max(map_in)#/2.      #(such that: map/norm will have max value 2)
        map_norm = map_in/norm
        map_poi = []
        for pix in range(len(map_in)):
            map_poi.append(np.random.poisson(map_norm[pix]))
        
        hp.write_map('map_poi.fits',np.array(map_poi))
        # exit()
        return np.array(map_poi)#*norm

    def rotation_gamma(self, nbase,rot_m_array,sin4psi,cos4psi ):
        #this can be translated with einsum
        npix_in = hp.nside2npix(self._nside_in)
        
        # gamma_rot[pol][pix]
        
        if self.pol == False: gamma_rot = [self.gammaI[nbase][rot_m_array]]
        else:
            gamma_rot = 4*[np.zeros(npix_in)]
            
            gamma_rot[0] = self.gammaI[nbase][rot_m_array]
            gamma_rot[1] = self.gammaV[nbase][rot_m_array] #1.j*((-1.j*self.gammaV[nbase])[rot_m_array])
            gamma_rot[2] = cos4psi*self.gammaQ[nbase][rot_m_array] - sin4psi*self.gammaU[nbase][rot_m_array]
            gamma_rot[3] = cos4psi*self.gammaU[nbase][rot_m_array] + sin4psi*self.gammaQ[nbase][rot_m_array]
            
        return gamma_rot[0:self.npol]
 #   def cutout(self,x, freqs,low = 20, high = 300):
    def simbase(self,freqs,q_n,pix_b,nbase,poi = False):
        
        npix_in = hp.nside2npix(self._nside_in)
        delta_freq = (freqs[1] - freqs[0])      #1.*self.fs/len(freqs)
        
        window = np.ones_like(freqs)    #might make sense with a window =/ box
        rot_m_array, sin4psi, cos4psi = self.rotation_pix(np.arange(npix_in), q_n)
        
        gamma_rot = self.rotation_gamma(nbase,rot_m_array,sin4psi,cos4psi)
        
        # hp.mollview(gammaI_rot)
        # plt.savefig('gammaI_rot.pdf')        
        # hp.mollview(gammaI_rot)
        # plt.savefig('gammaI_rot.pdf')
        
        vec_p_in = hp.pix2vec(self._nside_in,np.arange(npix_in))
        vec_b = hp.pix2vec(self._nside_in,pix_b)
        
        bdotp_in = 2.*np.pi*np.dot(vec_b,vec_p_in)*self.baseline_length[0]/3.e8        # edit ! self.R_earth
        
        df = np.zeros_like(freqs, dtype = complex)
        df1 = np.zeros_like(freqs, dtype = complex)
        
        map_in = self.map_in

        #maprot = np.einsum('ij,j->i',gamma_rot, map_in)
        
        if poi == True: 
            if self.pol == False: map_in = self.poissonify(map_in)
        #get map now: with poisson process with sigma = pixel in map_in
        
        # # jet = cm.jet
        # # jet.set_under("w")
        # hp.mollview(map_in)#,norm = 'hist', cmap = jet)
        # plt.savefig('test/map_poi_now.pdf' )
        # plt.close('all')
        #
        alpha = self.alpha
        f0 = self.f0
        
        for idx_f,f in enumerate(freqs):   
            
            exp_bdotp = (np.cos(bdotp_in*f) + np.sin(bdotp_in*f)*1.j)
            
            Apix = self.npol*[np.zeros_like(map_in[0], dtype = complex)]
            
            for i in range(len(Apix)):
                Apix[i] = map_in[i]*exp_bdotp            

            df[idx_f] = 4.*np.pi/npix_in * delta_freq*window[idx_f] * self.E_f(f,alpha,f0) * np.einsum('ij,ij->', gamma_rot, Apix )

            #print map_in[0][0]*sum(gamma_rot[0]*np.cos(bdotp_in*f)), map_in[0][0]*sum(gamma_rot[0]*np.sin(bdotp_in*f))
            #print -map_in[1][0]*sum(gamma_rot[1]*np.sin(bdotp_in*f)), map_in[1][0]*sum(gamma_rot[1]*np.cos(bdotp_in*f))
        return df
    
    def PDX(self,frexx,a,b,c):
        #b = 1.
        #c = 1.
         
        return (a*1.e-22*((18.*b)/(0.1+frexx))**2)**2+(0.07*a*1.e-22)**2+((frexx/(2000.*c))*.4*a*1.e-22)**2
    
    def Pdx_notcher(self,freqx,Pdx):
        mask = np.ones_like(freqx, dtype = bool)
        
        for (idx_f,f) in enumerate(freqx):
            for i in range(len(self.notches())):
                if f > (self.notches()[i]-15.*self.sigmas()[i]) and f < (self.notches()[i]+15.*self.sigmas()[i]):
                    mask[idx_f] = 0
                    
        
        # plt.figure()
        # plt.loglog(freqx[mask],Pdx[mask])
        # plt.savefig('masked.pdf')
        
        return freqx[mask],Pdx[mask]
    
    def noisy(self,strains_corr,psds_f,mask):
        psd_corr = []
        strains_noised = []
        
        for i in range(self._nbase):
            a, b = self.combo_tuples[i]
            psd_corr.append(psds_f[a]*psds_f[b])
            rands = [np.random.normal(loc = 0., scale = 1. , size = len(psd_corr[i])),np.random.normal(loc = 0., scale = 1. , size = len(psd_corr[i]))] 
            fakenoise = rands[0]+1.j*rands[1]
            fakenoise = np.array(fakenoise*np.sqrt(psd_corr[i]/2.))
            strain_noised = np.sum([fakenoise,strains_corr[i]], axis=0)
            strains_noised.append(strain_noised)
            
            # plt.figure()
            # plt.loglog(np.abs(fakenoise))
            # plt.savefig('noise%s.pdf' % i )
            # plt.close('all')
            #
            # plt.figure()
            # plt.loglog(np.abs(strain_noised))
            # plt.savefig('straincorr%s.pdf' % i )
            # plt.close('all')
            
        return strains_noised
    
    def injector(self,strains_in,ct_split,low_f,high_f,poi,sim = False):
        fs=self.fs        
        dt=1./fs
        
        ndects = self.ndet
        
        Nt = len(strains_in[0])
        Nt = lf.bestFFTlength(Nt)
        freqs = np.fft.rfftfreq(Nt, dt)
        
        mask = (freqs>low_f) & (freqs < high_f)
        mask2 = (freqs>80.) & (freqs < 300.)

        #print '+sim+'
    
        psds = []
        faketot = []
        fakestreams = []
        streams = []
        
        if sim == True:     #simulates bases for all detectors called when T.scope was initialised
            
            pix_bs = self.geometry(ct_split)[0]
            q_ns = self.geometry(ct_split)[1]
            
            for i in range(self._nbase):
                #a, b = self.combo_tuples[i]
                pix_b = pix_bs[i]
                q_x = q_ns[i]
                fakestream_corr = self.simbase(freqs[mask],q_x,pix_b,i,poi) 
                
                fakestreams.append(fakestream_corr)
            
            streams = fakestreams
        
        # if sim == False:
        #
        #     for i in range(len(strains_in)):
        #         streams.append(self.filter(strains_in[i], low_f,high_f))

        #**** psd ******
        flags = np.zeros(len(strains_in), dtype = bool)
        
        for (idx_str,strain_in) in enumerate(strains_in):
        
            '''WINDOWING & RFFTING.'''
            
            strain_in = strain_in[:Nt]
            strain_in_nowin = np.copy(strain_in)
            strain_in_nowin *= signal.tukey(Nt,alpha=0.05)
            strain_in *= np.blackman(Nt)

            hf = np.fft.rfft(strain_in, n=Nt, norm = 'ortho') 
            hf_nowin = np.fft.rfft(strain_in_nowin, n=Nt, norm = 'ortho') 
                
            '''the PSD. '''
            
            fstar = fs
            
            # plt.figure()
            # plt.loglog(freqs,np.abs(hf_nowin)**2)
            # plt.savefig('hf_nowin_inj.png' )
            #
            # exit()
            
            Pxx, frexx = mlab.psd(strain_in_nowin, Fs=fs, NFFT=2*fstar,noverlap=fstar/2,window=np.blackman(2*fstar),scale_by_freq=True)
            hf_psd = interp1d(frexx,Pxx)
            hf_psd_data = abs(hf_nowin.copy()*np.conj(hf_nowin.copy())) 
            
            #*******************
            
            #masxx = (frexx>low_f) & (frexx < (high_f+100.))

            if high_f < 300. or low_f<30.:
                masxx = (frexx>30.) & (frexx < 300.)
                                
            else:        
                masxx = (frexx>low_f) & (frexx < high_f)
            
            if int(high_f-low_f)<5: 
                masxx = (frexx>30.) & (frexx < 500.)
                
            frexx_cp = np.copy(frexx)
            Pxx_cp = np.copy(Pxx)
            frexx_cp = frexx_cp[masxx]
            Pxx_cp = Pxx_cp[masxx]
            
            #*******************

            frexx_notch,Pxx_notch = self.Pdx_notcher(frexx_cp,Pxx_cp)

            try:
                fit = curve_fit(self.PDX, frexx_notch, Pxx_notch)#, bounds = ([0.,0.,0.],[2.,2.,2.])) 
                psd_params = fit[0]
                
            except RuntimeError:
                print("Error - curve_fit failed")
                psd_params = [10.,10.,10.]
                
            
            
            a,b,c = psd_params
            min = 0.1
            max = 2.0
            
            #print psd_params
            
            if a < min or a > (max/2*1.5): flags[idx_str] = True
            if b < 2*min or b > 2*max: flags[idx_str] = True
            if c < 2*min or c > 12000*max: flags[idx_str] = True  # not drammatic if fit returns very high knee freq, ala the offset is ~1
            
            norm = np.mean(hf_psd_data[mask2])/np.mean(hf_psd(freqs)[mask2])#/np.mean(self.PDX(freqs,a,b,c))
                        
            if norm > 4500. : flags[idx_str] = True
            
            #if a < min or a > (max): flags[idx_str] = True
            #if c < 2*min or c > 2*max: flags[idx_str] = True  # not drammatic if fit returns very high knee freq, ala the offset is ~1

            
            if flags[idx_str] == True: print 'bad segment!  params', psd_params, 'ctime', ct_split[0]
            #Norm
            #
            # print np.mean(np.sqrt(hf_psd_data[mask]))/(abs(np.mean(np.real(hf_nowin)))+abs(np.mean(np.imag(hf_nowin))))
            #

            
            psd_params[0] = psd_params[0]*np.sqrt(norm) 
            a = a*np.sqrt(norm)      #ADD THE SQRT 2!!!!
            
            
            # s = int(ct_split[0])
            #
            # plt.figure()
            # plt.loglog(freqs[mask],hf_psd_data[mask], label = 'data')
            # #plt.loglog(freqs[mask],norm*hf_psd(freqs)[mask])
            # plt.loglog(freqs[mask],self.PDX(freqs,np.sqrt(norm),1.,1.)[mask], label = 'theo pdx fit')
            # plt.loglog(freqs[mask],self.PDX(freqs,a,b,c)[mask], label = 'notched pdx fit')
            # #plt.loglog(frexx_notch, norm*Pxx_notch, label = 'fittings')
            # plt.xlim(20.,1000.)
            # plt.legend()
            # plt.savefig('norm%s.pdf' % s)
            
            # if flags[idx_str] == True:
            #     s = int(ct_split[0])
            #
            #     plt.figure()
            #     plt.loglog(freqs[mask],norm*hf_psd(freqs)[mask])
            #     plt.loglog(freqs[mask],self.PDX(freqs,np.sqrt(norm),1.,1.)[mask], label = 'theo pdx fit')
            #     plt.loglog(freqs[mask],self.PDX(freqs,a,b,c)[mask], label = 'notched pdx fit')
            #     plt.loglog(frexx_notch, norm*Pxx_notch, label = 'fittings')
            #     plt.xlim(20.,1000.)
            #     plt.legend()
            #     plt.savefig('norm%s.pdf' % s)
            #
            # if flags[idx_str] == True: exit()
            
            psds.append(psd_params)
            #print frexx, Pxx, len(Pxx)
    
            #Pxx, frexx = mlab.psd(strain_in_win[:Nt], Fs = fs, NFFT = 4*fs, window = mlab.window_none)
            
            # plt.figure()
            # #plt.plot(freqs,) 
            # plt.savefig('.png' )
        
        lenpsds = len(psds)             #to fill in gaps if we are simulating        
        while ndects > lenpsds:
            psds.append(psds[0])
            lenpsds+=1
        
        if sim == False: return psds, flags
        if sim == True: return fakestreams, psds, flags
        
        
        ####
        
    def notches(self):
        
        if self.data_run == 'S6':
            notch_fs = np.array([14.0,34.70, 35.30, 35.90, 36.70, 37.30, 40.95, 60.00, 120.00, 179.99, 304.99, 331.49, 510.02, 1009.99])
        
        if self.data_run == 'O1':#34.70, 35.30,  #LIVNGSTON: 33.7 34.7 35.3 
            notch_fs = np.array([ 34.70, 35.30,35.90, 36.70, 37.30, 40.95, 60.00, 120.00, 179.99, 304.99, 331.9, 499.0, 500.0, 510.02,  1009.99])
        
        return notch_fs
        
    def sigmas(self):
        
        if self.data_run == 'S6':
            sigma_fs = np.array([.5,.5,.5,.5,.5,.5,.5,1.,1.,1.,1.,5.,1.,1.])
        
        if self.data_run == 'O1':
            sigma_fs = np.array([.5,.5,.5,.5,.5,.5,.5,1.,1.,1.,1.,2.,2.,2.,1.])    
            
        return sigma_fs
        #np.array([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.5,0.3,0.2])
    
    def filter(self,strain_in,low_f,high_f,simulate = False):
        fs=self.fs        
        dt=1./fs
        
        '''WINDOWING & RFFTING.'''
        
        Nt = len(strain_in)
        Nt = lf.bestFFTlength(Nt)
        
        strain_in = strain_in[:Nt]
        strain_in_cp = np.copy(strain_in)
        strain_in_nowin = np.copy(strain_in)
        strain_in_nowin *= signal.tukey(Nt,alpha=0.05)
        #strain_in *= np.blackman(Nt)
        freqs = np.fft.rfftfreq(Nt, dt)
        #print '=rfft='
        hf_nowin = np.fft.rfft(strain_in_nowin, n=Nt, norm = 'ortho') #####!HERE! 03/03/18 #####
        
        
        hf_copy = np.copy(hf_nowin)
        
        
        #Norm
        mask = (freqs>low_f) & (freqs < high_f)
        
        
        '''NOTCHING. '''
        
        notch_fs = self.notches()
        sigma_fs = self.sigmas()
        #np.array([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.5,0.3,0.2])
        
        df = freqs[1]- freqs[0]
        samp_hz = 1./df
        pixels = np.arange(len(hf_copy))
             
        i = 0
         
        hf_ones = np.ones_like(hf_nowin)
         
        while i < len(notch_fs):
            notch_pix = int(notch_fs[i]*samp_hz)
            hf_ones = hf_ones*(1.-self.gaussian(pixels,notch_pix,sigma_fs[i]*samp_hz))
            hf_nowin = hf_nowin*(1.-self.gaussian(pixels,notch_pix,sigma_fs[i]*samp_hz))
            i+=1           
        
        # plt.figure()
        # plt.loglog(freqs[mask],np.abs(hf_nowin[mask])**2)
        # plt.savefig('hf_notchin.png' )
        #BPING HF
        gauss_lo = self.halfgaussian(pixels,low_f*samp_hz,samp_hz)
        gauss_hi = self.halfgaussian(pixels,high_f*samp_hz,samp_hz)

        hf_nbped = hf_nowin*(1.-gauss_lo)*(gauss_hi)            ####
        
        return hf_nowin#, hf_psd
        

    # ********* Data Segmenter *********

    def flagger(self,start,stop,filelist):
        
        fs = self.fs
        # convert LIGO GPS time to datetime
        # make sure datetime object knows it is UTC timezone
        utc_start = tconvert(start).replace(tzinfo=pytz.utc)
        utc_stop = tconvert(stop).replace(tzinfo=pytz.utc)

        # 1970-1-1 in UTC defines epoch of unix time 
        epoch = dt.datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)

        #print (utc_start - epoch).total_seconds()
        #print (utc_stop - epoch).total_seconds()

        # get segments with required flag level
        segs_H1_cat1 = rl.getsegs(start, stop, 'H1',flag = 'CBC_CAT1', filelist=filelist)   #'STOCH_CAT1'
        inj_H1_detc = rl.getsegs(start, stop, 'H1',flag = 'NO_DETCHAR_HW_INJ', filelist=filelist) 
        inj_H1_stoch = rl.getsegs(start, stop, 'H1',flag = 'NO_STOCH_HW_INJ', filelist=filelist)
        
        inj_H1_stoch = segmentlist(inj_H1_stoch)
        inj_H1_detc = segmentlist(inj_H1_detc)
        segs_H1_cat1 = segmentlist(segs_H1_cat1)
        
        segs_H1 = inj_H1_stoch&inj_H1_detc&segs_H1_cat1
        
        good_data_H1 = np.zeros(stop-start,dtype=np.bool)
        for (begin, end) in segs_H1:
            good_data_H1[begin-start:end-start] = True

        #segs_L1 = rl.getsegs(start, stop, 'L1',flag = 'CBC_CAT1', filelist=filelist)  #flag='STOCH_CAT1',
        
        segs_L1_cat1 = rl.getsegs(start, stop, 'L1',flag = 'CBC_CAT1', filelist=filelist)   #'STOCH_CAT1'
        inj_L1_detc = rl.getsegs(start, stop, 'L1',flag = 'NO_DETCHAR_HW_INJ', filelist=filelist) 
        inj_L1_stoch = rl.getsegs(start, stop, 'L1',flag = 'NO_STOCH_HW_INJ', filelist=filelist)
        
        inj_L1_stoch = segmentlist(inj_L1_stoch)
        inj_L1_detc = segmentlist(inj_L1_detc)
        segs_L1_cat1 = segmentlist(segs_L1_cat1)
        
        segs_L1 = inj_L1_stoch&inj_L1_detc&segs_L1_cat1
        
        good_data_L1 = np.zeros(stop-start,dtype=np.bool)
        for (begin, end) in segs_L1:
            good_data_L1[begin-start:end-start] = True

        # add time bit at beginning and end to _AND_ of two timeseries
        good_data = np.append(np.append(False,good_data_H1 & good_data_L1),False)
        # do diff to identify segments
        diff = np.diff(good_data.astype(int))
        segs_begin = np.where(diff>0)[0] + start #+1
        segs_end =  np.where(diff<0)[0] + start #+1

        # re-define without first and last time bit
        # This mask now defines conincident data from both L1 and H1
        good_data = good_data_H1 & good_data_L1

        # TODO: Add flagging of injections

        # Now loop over all segments found
        
        return segs_begin, segs_end 
        
        #for sdx, (begin, end) in enumerate(zip(segs_begin,segs_end)):    

    def segmenter(self, begin, end, filelist):
        fs = self.fs
        # load data
        strain_H1, meta_H1, dq_H1 = rl.getstrain(begin, end, 'H1', filelist=filelist)
        strain_L1, meta_L1, dq_L1 = rl.getstrain(begin, end, 'L1', filelist=filelist)

        print '+++'
        epoch = dt.datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
        print '+++'
        
        # Figure out unix time for this segment
        # This is the ctime for qpoint
        utc_begin = tconvert(meta_H1['start']).replace(tzinfo=pytz.utc)
        utc_end = tconvert(meta_H1['stop']).replace(tzinfo=pytz.utc)
        ctime = np.arange((utc_begin - epoch).total_seconds(),(utc_end - epoch).total_seconds(),meta_H1['dt'])

        # discard very short segments
        if len(ctime)/fs < 16: 
            return [0.],[0.],[0.]
        print utc_begin, utc_end

        # if long then split into sub segments
        if len(ctime)/fs > 120:
            # split segment into sub parts
            # not interested in freq < 40Hz
            n_split = np.int(len(ctime)/(60*fs))
            print 'split ', n_split, len(ctime)/fs  ############
            ctime_seg = np.array_split(ctime, n_split)
            strain_H1_seg = np.array_split(strain_H1, n_split)
            strain_L1_seg = np.array_split(strain_L1, n_split)
            #step=int(len(strain_H1_seg[0])/2)
            #ctime_over = np.array_split(ctime[step:-step], n_split-1)
            #strain_H1_over = np.array_split(strain_H1[step:-step], n_split-1)
            #strain_L1_over = np.array_split(strain_L1[step:-step], n_split-1)
            
            #zipped_ct = np.array(zip(ctime_seg[:-1],ctime_over))
            #print zipped_ct
            #print len(zipped_ct),len(zipped_ct[0]), len(zipped_ct[0][0])
            #ctime_out = zipped_ct.reshape((2*len(ctime_over),len(ctime_seg[0])))
            
            #print len(ctime_seg), len(ctime_seg[1])
            
            #zipped_h = np.array(zip(strain_H1_seg[:-1],strain_H1_over))
            #strain_H1_out = zipped_h.reshape(-1, zipped_h.shape[-1])
            
            #zipped_l = np.array(zip(strain_L1_seg[:-1],strain_L1_over))
            #strain_L1_out = zipped_l.reshape(-1, zipped_l.shape[-1])
            
            #while len(strain_H1_seg[-1])<len(strain_H1_seg[0]):
            #    strain_H1_seg[-1] = np.append(strain_H1_seg[-1],0.)
            #    strain_L1_seg[-1] = np.append(strain_L1_seg[-1],0.)
            #    ctime_seg[-1] = np.append(ctime_seg[-1],0.)
            
            
            #strain_H1_out = np.vstack((strain_H1_out,strain_H1_seg[-1]))
            #strain_L1_out = np.vstack((strain_L1_out,strain_L1_seg[-1]))
            #ctime_out = np.vstack((ctime_out,ctime_seg[-1]))
            
            #print len(zipped_h),len(zipped_h[0]), len(zipped_h[0][0])
            #print len(strain_H1_out), len(strain_H1_out[0])
            #print len(zipped_ct),len(zipped_ct[0]), len(zipped_ct[0][0])
            #print len(ctime_out), len(ctime_out[0])
                        
            return ctime_seg, strain_H1_seg, strain_L1_seg #strain_x
            
        else:
            # add dummy dimension
            n_split = 1
            ctime = ctime[None,...]
            strain_H1 = strain_H1[None,...]
            strain_L1 = strain_L1[None,...]
            return ctime, strain_H1, strain_L1 #strain_x
    

    # ********* Projector *********
    # returns p = {lm} map of inverse-noise-filtered time-stream

    def summer(self, ct_split, strains, pows, freq, pix_b, q_n , norm):     
        
        npol = self.npol
        
        nside=self._nside_out
        
        # print len(freq), max(freq)
        # print 4096./len(freq)
                
        mask = (freq>self.low_f) & (freq < self.high_f)
        freq = freq[mask]
        window = np.ones_like(freq)
        
        alpha = self.alpha
        f0 = self.f0
        
        #delf = self.fs/float(len(freq))#/len(strain[0]) #self.fs/4./len(strain[0]) SHOULD TAKE INTO ACCOUNT THE *2, THE NORMALISATION (1/L) AND THE DELTA F
        #geometry 
        delf = (freq[1]-freq[0])    ##/(self.high_f-self.low_f) 
        
        # print delf, freq[1]-freq[0]
        
        npix_out = self.npix_out
        npix_in = self.npix_in
        
        mid_idx = int(len(ct_split)/2)
    
        vec_p_out = hp.pix2vec(self._nside_out,np.arange(npix_out))
        
        #test?
        
        z_p = np.zeros((npix_out,npol),dtype = complex)
        M_pp = np.zeros((npix_out,npix_out,npol,npol),dtype = complex)
        
        for idx_b in range(self._nbase):
            
            #print idx_b
            
            rot_m_array, sin4psi, cos4psi = self.rotation_pix(np.arange(npix_in), q_n[idx_b])  
            gamma_rot = self.rotation_gamma(idx_b,rot_m_array,sin4psi, cos4psi)
            
            #hp.fitsfunc.write_map('gammaI_rot.fits', gammaI_rot) 
            
            
            gamma_rot_ud = np.transpose(hp.ud_grade(gamma_rot,nside_out = self._nside_out)) 
            
            #print gamma_rot_ud, len(gamma_rot_ud), len(gamma_rot_ud[0])
            #print 'bpix:' , pix_b[idx_b]
            
            vec_b = hp.pix2vec(self._nside_in,pix_b[idx_b])
            
            bdotp = 2.*np.pi*np.dot(vec_b,vec_p_out)*self.baseline_length[0]/3.e8  #self.R_earth   edit !
                        
            df = strains[idx_b]
            pf = pows[idx_b]#[mask]
        
            
            Ef = self.E_f(freq,alpha,f0)

            #for pol_idx in range(4):             
            for ip in range(npix_out):
                
                if npol == 1:
                                
                    z_p[ip] += 8.*np.pi/npix_out * delf* gamma_rot_ud[ip]*np.sum(window[:] 
                                * Ef[:]/ pf[:]      ## minus sign? changed it to +
                                *(np.cos(bdotp[ip]*freq[:])*np.real(df[:]) + np.sin(bdotp[ip]*freq[:])*np.imag(df[:]))) 
                    
                    
                    
                elif npol == 2:
                                        
                    z_p[ip][0] += 8.*np.pi/npix_out * delf* gamma_rot_ud[ip][0]*np.sum(window[:] 
                                * Ef[:]/ pf[:]      ## minus sign? changed it to +
                                *(np.cos(bdotp[ip]*freq[:])*np.real(df[:]) + np.sin(bdotp[ip]*freq[:])*np.imag(df[:])))
                
                    z_p[ip][1] += 8.*np.pi/npix_out * delf* 1.j* gamma_rot_ud[ip][1]*np.sum(window[:] 
                                * Ef[:]/ pf[:]      ## minus sign? changed it to +
                                *(np.cos(bdotp[ip]*freq[:])*np.imag(df[:]) - np.sin(bdotp[ip]*freq[:])*np.real(df[:]) ))
                
                for jp in range(ip,npix_out):
                    
                    #val = einsum(W,W' -> WW')      sym?
                    if npol == 1:     
                        val = np.einsum('i,j -> ij', 2.*(4.*np.pi)**2/npix_out**2 * delf**2 * gamma_rot_ud[ip] * np.sum(window[:]**2 * Ef[:]**2/ pf[:]*(np.cos((bdotp[ip]-bdotp[jp])*freq[:]))), gamma_rot_ud[jp])
                    
                        M_pp[ip,jp] += val
                        
                        if ip!= jp : M_pp[jp,ip] += val       
                    
                    if npol == 2:
                        
                        const = 2.*(4.*np.pi)**2/npix_out**2 * delf**2
                        val1 = const * gamma_rot_ud[ip][0] * np.sum(window[:]**2 * Ef[:]**2/ pf[:]*(np.cos((bdotp[ip]-bdotp[jp])*freq[:]) )) * gamma_rot_ud[jp][0]
                        val2 = const * 1.j*gamma_rot_ud[ip][0] * np.sum(window[:]**2 * Ef[:]**2/ pf[:]*(np.sin((bdotp[ip]-bdotp[jp])*freq[:]) )) * np.conj(gamma_rot_ud[jp][1]) 
                        val3 = const * 1.j*gamma_rot_ud[jp][0] * np.sum(window[:]**2 * Ef[:]**2/ pf[:]*(np.sin((bdotp[ip]-bdotp[jp])*freq[:]) )) *gamma_rot_ud[ip][1]
                        val4 = const * gamma_rot_ud[ip][1] * np.sum(window[:]**2 * Ef[:]**2/ pf[:]*(np.cos((bdotp[ip]-bdotp[jp])*freq[:]) )) * np.conj(gamma_rot_ud[jp][1])  
                        
                        val =[[val1,val2],[val2,val4]]
                        
                        M_pp[ip,jp] += val

                        if ip!= jp :
                        
                            const = 2.*(4.*np.pi)**2/npix_out**2 * delf**2
                            val1 = const * gamma_rot_ud[ip][0] * np.sum(window[:]**2 * Ef[:]**2/ pf[:]*(np.cos((bdotp[jp]-bdotp[ip])*freq[:]) )) * gamma_rot_ud[jp][0]
                            val2 = const * 1.j*gamma_rot_ud[jp][0] * np.sum(window[:]**2 * Ef[:]**2/ pf[:]*(np.sin((bdotp[jp]-bdotp[ip])*freq[:]) )) * np.conj(gamma_rot_ud[ip][1]) 
                            val3 = const * 1.j*gamma_rot_ud[ip][0] * np.sum(window[:]**2 * Ef[:]**2/ pf[:]*(np.sin((bdotp[jp]-bdotp[ip])*freq[:]) )) *gamma_rot_ud[jp][1]
                            val4 = const * gamma_rot_ud[jp][1] * np.sum(window[:]**2 * Ef[:]**2/ pf[:]*(np.cos((bdotp[jp]-bdotp[ip])*freq[:]) )) * np.conj(gamma_rot_ud[ip][1])  
                        
                            val =[[val1,val2],[val2,val4]]
                        
                            M_pp[jp,ip] += [[val1,val2],[val2,val4]]     
             
        # plt.figure()
        #
        # plt.loglog(freq, np.abs(df)**2, label = 'df df*')
        # plt.loglog(freq, pf, label = 'P1P2')
        #
        # plt.legend()
        # plt.savefig('compare.png' )
        #
        # print np.mean(np.abs(df)**2), np.mean(pf)
        # print np.average(np.abs(df)**2), np.average(pf)
        
        return z_p, M_pp 


    def projector(self,ctime, s, psds, freqs,pix_bs, q_ns, norm = False):
        
        #just a summer wrapper really
        
        #print 'proj run'
            
        nside=self._nside_out
        
        pows = []
        
        for i in range(self._nbase):
            a, b = self.combo_tuples[i]
            pows.append(psds[a]*(psds[b]))      #(or sqrt)
                
        z_p, M_pp = self.summer(ctime, s, pows, freqs, pix_bs, q_ns, norm)
        
        #print np.mean(data_lm)          
        
        return z_p, M_pp
        

    # ********* S(f) *********

    #def S(self, s, freqs):
        
        #return 
