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
import stokefields as sfs
from numpy import cos,sin
from matplotlib import cm


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
    
    if noise_lvl == 1: alpha = 1.
    elif noise_lvl == 2: alpha = 1.e-38
    elif noise_lvl == 3: alpha = 3.e-39
    elif noise_lvl == 4: alpha = 5.e-40
    elif noise_lvl == 5: alpha = 1.e-41

    
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
    
    return (np.vstack(hp.synfast(cls, nside=nside, pol=True, new=True)).flatten())*alpha

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
        
        Istoke = hp.sphtfunc.alm2map(self.a_lm, nside)
            
            
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

    def __init__(self, nside_in,nside_out, lmax, fs, low_f, high_f, dects, maptyp, this_path, noise_lvl=1, data_run = 'O1'): #Dect list
    
        self.Q = qp.QPoint(accuracy='low', fast_math=True, mean_aber=True)#, num_threads=1)
        
        self.this_path = this_path
        self.data_run = data_run
        
        self.R_earth = 6378137
        self._nside_in = nside_in
        self._nside_out = nside_out
        self._lmax = lmax
        self.fs = fs
        self.low_f = low_f
        self.high_f = high_f
        
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
                
        np.savez('baseline_lengths.npz', baseline_length = self.baseline_length, dects = dects, combos = self.combo_tuples )
        print 'saved baseline_lengths.npz'
        
        self.npix_in = hp.nside2npix(self._nside_in)
        self.npix_out = hp.nside2npix(self._nside_out)

        # calculate overlap functions
        # TODO: integrate this with general detector table
        #theta, phi = hp.pix2ang(self._nside,np.arange(self.npix)) 
        
        self.gammaI = []
        
        for i in range(self._nbase):
            a, b = self.combo_tuples[i]
            self.gammaI.append((5./(8.*np.pi))*self.detectors[a].get_Fplus()*self.detectors[b].get_Fplus()+self.detectors[a].get_Fcross()*self.detectors[b].get_Fcross())
                
        # hp.mollview(self.gammaI[0])
        # plt.savefig('gammaI.pdf')

        #self.gammaQ = self.H1.get_Fplus()*self.L1.get_Fcross()-self.H1.get_Fcross()*self.L1.get_Fplus()
        #self.gammaU = self.H1.get_Fplus()*self.L1.get_Fplus()-self.H1.get_Fcross()*self.L1.get_Fcross()
        #self.gammaV = self.H1.get_Fplus()*self.L1.get_Fcross()+self.H1.get_Fcross()*self.L1.get_Fplus()
        
        
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
        
        #Simulation tools
    
        self.noise_lvl = noise_lvl
        self.alpha = 1.

        if noise_lvl == 1: alpha = 1.
        elif noise_lvl == 2: alpha = 1.e-38
        elif noise_lvl == 3: alpha = 3.e-39
        elif noise_lvl == 4: alpha = 5.e-40
        elif noise_lvl == 5: alpha = 1.e-41
        
        
        self.alpha = alpha
        
        print 'alpha is', self.alpha

        input_map = self.get_map_in(maptyp)
        self.map_in = input_map.copy()
        

        # plt.figure()
        # hp.mollview(self.map_in)
        # plt.savefig('map_in.pdf' )
        
    # ********* Maps IN *************
    
    def get_map_in(self, maptyp):
        
        nside = self._nside_in
        alpha = self.alpha
        
        lmax = nside/2        #or not?

        alm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=np.complex)
        
        if maptyp == None: 
            map_in = hp.alm2map(alm,nside=self._nside_in)
            
        elif maptyp == '1pole':
            idx = hp.Alm.getidx(lmax,0,0)
            alm[idx] = (1.+ 0.j)*alpha

            map_in = hp.alm2map(alm,nside=self._nside_in)
        
        elif maptyp == '2pole':
            idx = hp.Alm.getidx(lmax,1,1)
            alm[idx] = (1.+ 0.j)*alpha
            
            map_in = hp.alm2map(alm,nside=self._nside_in)

        elif maptyp == '2pole1':
            idx = hp.Alm.getidx(lmax,1,0)
            alm[idx] = (1.+ 0.j)*alpha
            
            idx = hp.Alm.getidx(lmax,1,1)
            alm[idx] = (.58+ .73j)*alpha
            #idx = hp.Alm.getidx(lmax,1,1)
            #alm[idx] = 1.+ 0.j
            
            map_in = hp.alm2map(alm,nside=self._nside_in)
        
        elif maptyp == '4pole':
            idx = hp.Alm.getidx(lmax,2,2)
            alm[idx] = (1.+ 0.j)*alpha
            
            map_in = hp.alm2map(alm,nside=self._nside_in)
            
        elif maptyp == '8pole':
            idx = hp.Alm.getidx(lmax,3,3)
            alm[idx] = (1.+ 0.j)*alpha
            
            map_in = hp.alm2map(alm,nside=self._nside_in)
        
        elif maptyp == '8pole1':
            idx = hp.Alm.getidx(lmax,3,3)
            alm[idx] = (1.+ .72j)*alpha
            idx = hp.Alm.getidx(lmax,3,2)
            alm[idx] = (.58+ .67j)*alpha
            
            map_in = hp.alm2map(alm,nside=self._nside_in)
        
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
            planckmap = hp.sphtfunc.smoothing(planckmap,fwhm = fwhm)
            map_in = (hp.ud_grade(planckmap,nside_out = self._nside_in))
            max_in = max(map_in)
            map_in = map_in/max_in*alpha
        
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
            
        quatmap_rot_pix = self.Q.quat2pix(quatmap_rotated,nside)[0] #rotated pixel list (polarizations are in [1])
        return quatmap_rot_pix

    def E_f(self,f,alpha=3.,f0=1.):
        return (f/f0)**(alpha-3.)
    
    def coupK(self,l,lp,lpp,m,mp):
        return np.sqrt((2*l+1.)*(2*lp+1.)*(2*lpp+1.)/4./np.pi)*self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp]

    def dfreq_factor(self,f,ell,idx_base,H0=68.0,f0=100.):
        # f : frequency (Hz)
        # ell : multipole
        # alpha: spectral index
        # b: baseline length (m)
        # f0: pivot frequency (Hz)
        # H0: Hubble rate today (km/s/Mpc)
        
        b=self.baseline_length[idx_base]
        

        km_mpc = 3.086e+19 # km/Mpc conversion
        c = 3.e8 # speed of light 
        #fac = 8.*np.pi**3/3./H0**2 * km_mpc**2 * f**3*(f/f0)**(alpha-3.) * spherical_jn(ell, 2.*np.pi*f*b/c)
        fac =  spherical_jn(ell, 2.*np.pi*(f)*b/c)*self.E_f(f)
        # add band pass and notches here
        
        return fac

    def freq_factor(self,ell,alpha=3.,H0=68.0,f0=100.):
        fmin=self.low_f
        fmax=self.high_f
        return integrate.quad(self.dfreq_factor,fmin,fmax,args=(ell))[0]
    
    def vec2azel(self,v1,v2):
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
        print 'nside for bpix is: ' , nside
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

    def iir_bandstops(self, fstops, fs, order=4):
        """ellip notch filter
        fstops is a list of entries of the form [frequency (Hz), df, df2f]                           
        where df is the pass width and df2 is the stop width (narrower                              
        than the pass width). Use caution if passing more than one freq at a time,                  
        because the filter response might behave in ways you don't expect.
        """
        nyq = 0.5 * fs

        # Zeros zd, poles pd, and gain kd for the digital filter
        zd = np.array([])
        pd = np.array([])
        kd = 1

        # Notches
        for fstopData in fstops:
            fstop = fstopData[0]
            df = fstopData[1]
            df2 = fstopData[2]
            low = (fstop - df) / nyq
            high = (fstop + df) / nyq
            low2 = (fstop - df2) / nyq
            high2 = (fstop + df2) / nyq
            z, p, k = iirdesign([low,high], [low2,high2], gpass=1, gstop=6,
                                ftype='ellip', output='zpk')
            zd = np.append(zd,z)
            pd = np.append(pd,p)

        # Set gain to one at 100 Hz...better not notch there                                        
        bPrelim,aPrelim = zpk2tf(zd, pd, 1)
        outFreq, outg0 = freqz(bPrelim, aPrelim, 100/nyq)

        # Return the numerator and denominator of the digital filter                                
        b,a = zpk2tf(zd,pd,k)
        return b, a
    
    def get_filter_coefs(self, fs, bandpass=False):
    
        # assemble the filter b,a coefficients:
        coefs = []

        # bandpass filter parameters
        lowcut=20 #43
        highcut=300 #260
        order = 4

        # Frequencies of notches at known instrumental spectral line frequencies.
        # You can see these lines in the ASD above, so it is straightforward to make this list.
        notchesAbsolute = np.array([14.0,34.70, 35.30, 35.90, 36.70, 37.30, 40.95, 60.00, 120.00, 179.99, 304.99, 331.49, 510.02, 1009.99])
        # exclude notch below lowcut
        notchesAbsolute = notchesAbsolute[notchesAbsolute > lowcut]

        # notch filter coefficients:
        for notchf in notchesAbsolute:                      
            bn, an = self.iir_bandstops(np.array([[notchf,1,0.1]]), fs, order=4)
            coefs.append((bn,an))

        # Manually do a wider notch filter around 510 Hz etc.          
        bn, an = self.iir_bandstops(np.array([[510,200,20]]), fs, order=4)
        #coefs.append((bn, an))

        # also notch out the forest of lines around 331.5 Hz
        bn, an = self.iir_bandstops(np.array([[331.5,10,1]]), fs, order=4)
        #coefs.append((bn, an))

        if bandpass:
            # bandpass filter coefficients
            # do bandpass as last filter
            nyq = 0.5*fs
            low = lowcut / nyq
            high = highcut / nyq
            bb, ab = butter(order, [low, high], btype='band')
            coefs.append((bb,ab))
    
        return coefs

    def filter_data(self, data_in,coefs):
        data = data_in.copy()
        for coef in coefs:
            b,a = coef
            # filtfilt applies a linear filter twice, once forward and once backwards.
            # The combined filter has linear phase.
            data = filtfilt(b, a, data)
        return data
        

    def poissonify(self,map_in):
        norm = max(map_in)#/2.      #(such that: map/norm will have max value 2)
        map_norm = map_in/norm
        map_poi = []
        for pix in range(len(map_in)):
            map_poi.append(np.random.poisson(map_norm[pix]))
        
        # hp.write_map('map_poi.fits',np.array(map_poi))
        # exit()
        return np.array(map_poi)#*norm
        
 #   def cutout(self,x, freqs,low = 20, high = 300):
    def simbase(self,freqs,q_n,pix_b,nbase,poi = False):
        
        npix_in = hp.nside2npix(self._nside_in)
        delta_freq = 1.*self.fs/len(freqs)
        window = np.ones_like(freqs)    #might make sense with a window =/ box
        rot_m_array = self.rotation_pix(np.arange(npix_in), q_n)
        gammaI_rot = self.gammaI[nbase][rot_m_array]
        
        # hp.mollview(gammaI_rot)
        # plt.savefig('gammaI_rot.pdf')
        
        vec_p_in = hp.pix2vec(self._nside_in,np.arange(npix_in))
        vec_b = hp.pix2vec(self._nside_in,pix_b)
        bdotp_in = 2.*np.pi*np.dot(vec_b,vec_p_in)*self.R_earth/3.e8
        
        df = np.zeros_like(freqs, dtype = complex)
        
        map_in = self.map_in
        if poi == True: map_in = self.poissonify(map_in)
        #get map now: with poisson process with sigma = pixel in map_in
        
        # # jet = cm.jet
        # # jet.set_under("w")
        # hp.mollview(map_in)#,norm = 'hist', cmap = jet)
        # plt.savefig('test/map_poi_now.pdf' )
        # plt.close('all')
        #
        
        for idx_f,f in enumerate(freqs):     #maybe E_f is squared?
            df[idx_f] = 4.*np.pi/npix_in * delta_freq*np.sum(window[idx_f] * self.E_f(f) * gammaI_rot[:] * map_in[:]*(np.cos(bdotp_in[:]*f) + np.sin(bdotp_in[:]*f)*1.j)) 
        
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
            psd_corr.append(psds_f[a][mask]*psds_f[b][mask])
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
        freqs = np.fft.rfftfreq(2*Nt, dt)
        freqs = freqs[:Nt/2+1]
        
        mask = (freqs>low_f) & (freqs < high_f)

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
        
        if sim == False:
            
            for i in range(len(strains_in)):
                streams.append(self.filter(strains_in[i], low_f,high_f))

        #**** psd ******
        flags = np.zeros(len(strains_in), dtype = bool)
        
        for (idx_str,strain_in) in enumerate(strains_in):
        
            '''WINDOWING & RFFTING.'''
            
            strain_in = strain_in[:Nt]
            strain_in_nowin = np.copy(strain_in)
            strain_in_nowin *= signal.tukey(Nt,alpha=0.05)
            strain_in *= np.blackman(Nt)

            hf = np.fft.rfft(strain_in, n=2*Nt)#, norm = 'ortho') 
            hf_nowin = np.fft.rfft(strain_in_nowin, n=2*Nt)#, norm = 'ortho') 
    
            hf = hf[:Nt/2+1]
            hf_nowin = hf_nowin[:Nt/2+1]
            
            '''the PSD. '''
            
            fstar = fs
    
            Pxx, frexx = mlab.psd(strain_in_nowin, Fs=fs, NFFT=2*fstar,noverlap=fstar/2,window=np.blackman(2*fstar),scale_by_freq=True)
            hf_psd = interp1d(frexx,Pxx)
            hf_psd_data = abs(hf_nowin.copy()*np.conj(hf_nowin.copy())/(fs**2)) 
            
            #*******************

            frexx_notch,Pxx_notch = self.Pdx_notcher(frexx,Pxx)

            try:
                fit = curve_fit(self.PDX, frexx_notch[100:600], Pxx_notch[100:600])#, bounds = ([0.,0.,0.],[2.,2.,2.])) 
                psd_params = fit[0]
                
            except RuntimeError:
                print("Error - curve_fit failed")
                psd_params = [10.,10.,10.]
                
            
            
            a,b,c = psd_params
            min = 0.1
            max = 1.9
            
            #print psd_params
            
            if a < min or a > max: flags[idx_str] = True
            if b < 2*min or b > 2*max: flags[idx_str] = True
            if c < 2*min or c > 2*max: flags[idx_str] = True
            
            if flags[idx_str] == True: print 'bad segment!  '
            #Norm
            norm = np.mean(hf_psd_data[mask])/np.mean(hf_psd(freqs)[mask])#/np.mean(self.PDX(freqs,a,b,c))
            psd_params[0] = norm*psd_params[0] 
            
            
            # plt.figure()
            # plt.loglog(hf_psd_data[mask])
            # plt.loglog(hf_psd(freqs)[mask])
            # #plt.loglog(self.PDX(frexx,a,b,c)[mask], label = 'notched pdx fit')
            # plt.legend()
            # plt.savefig('norm.pdf' )
            #
            # print norm
            
            # plt.figure()
            # plt.loglog(frexx,Pxx,label = 'PSD from data')
            # plt.loglog(frexx,self.PDX(frexx,a,b,c),'r--',label = 'Fitted function')
            # #plt.loglog(frexxes,pixxes*1e-48)
            # #plt.loglog(frexx,interp1d(frexxes,pixxes)(frexx)*10E-47)
            # plt.axvline(x=self.notches()[0],linewidth = 0.5,label = 'Frequency notches')
            # for xc in self.notches()[1:]:
            #     plt.axvline(x=xc,linewidth = 0.5)
            # plt.legend()
            # plt.xlabel('frequency (Hz)')
            # plt.ylabel('PSD')
            # plt.xlim((1,1500))
            # plt.axvspan(low_f,high_f, alpha=0.5, color='#FFF700', zorder=-11)
            # # plt.ylim((1.e-47,1.e-43))
            # plt.savefig('PSD.pdf' )

            #hf_psd=interp1d(frexx,Pxx*norm)
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
            notch_fs = np.array([ 34.70, 35.30,35.90, 36.70, 37.30, 40.95, 60.00, 120.00, 179.99, 304.99, 331.9, 510.02,  1009.99])
        
        return notch_fs
        
    def sigmas(self):
        
        if self.data_run == 'S6':
            sigma_fs = np.array([.5,.5,.5,.5,.5,.5,.5,1.,1.,1.,1.,5.,5.,1.])
        
        if self.data_run == 'O1':
            sigma_fs = np.array([.5,.5,.5,.5,.5,.5,.5,1.,1.,1.,1.,5.,5.,1.])    
            
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
        strain_in_nowin *= signal.tukey(Nt,alpha=0.0001)
        #strain_in *= np.blackman(Nt)
        freqs = np.fft.rfftfreq(2*Nt, dt)
        #print '=rfft='
        hf_nowin = np.fft.rfft(strain_in_nowin, n=2*Nt, norm = 'ortho') #####!HERE! 03/03/18 #####
        
        hf_nowin = hf_nowin[:Nt/2+1]
        freqs = freqs[:Nt/2+1]
        
        #hf_back =  np.fft.irfft(hf_nowin, norm = 'ortho')
        #print np.average(hf_back), ' , ' , np.std(hf_back), '  ,  ', len(hf_back)
        
        
        hf_copy = np.copy(hf_nowin)
        
        #print '++'
        
                
        '''the PSD. '''
        #Pxx, frexx = mlab.psd(strain_in_nowin, Fs=fs, NFFT=2*fs,noverlap=fs/2,window=np.blackman(2*fs),scale_by_freq=True)
        #hf_psd = interp1d(frexx,Pxx)
        #hf_psd_data = abs(hf_nowin.copy()*np.conj(hf_nowin.copy())/(fs**2))
        
        #if sim: return simulated noise
        # strain_in = sim noise
        
        #Norm
        mask = (freqs>low_f) & (freqs < high_f)
        #norm = np.mean(hf_psd_data[mask])/np.mean(hf_psd(freqs)[mask])
        
        #print norm
        
        #hf_psd=interp1d(frexx,Pxx*norm)
        
        
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
        
        
        #BPING HF

        gauss_lo = self.halfgaussian(pixels,low_f*samp_hz,samp_hz)
        gauss_hi = self.halfgaussian(pixels,high_f*samp_hz,samp_hz)

        hf_nbped = hf_nowin*(1.-gauss_lo)*(gauss_hi)            ####
        
        mask1 = (freqs>30.) & (freqs < 1000.)
        
        # plt.figure()
        # plt.plot(freqs[mask1],np.abs(hf_copy[mask1]))
        # plt.loglog(freqs[mask1],np.abs(hf_nowin[mask1]))
        # for xc in self.notches():
        #     plt.axvline(x=xc,linewidth = 0.5)
        # plt.savefig('hf_nbped.png' )
        # plt.close('all')
        
        return hf_nbped#, hf_psd
        

    # ********* Data Segmenter *********

    def flagger(self,start,stop,filelist):
        
        fs = self.fs
        # convert LIGO GPS time to datetime
        # make sure datetime object knows it is UTC timezone
        utc_start = tconvert(start).replace(tzinfo=pytz.utc)
        utc_stop = tconvert(stop).replace(tzinfo=pytz.utc)

        # 1970-1-1 in UTC defines epoch of unix time 
        epoch = dt.datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)

        print (utc_start - epoch).total_seconds()
        print (utc_stop - epoch).total_seconds()

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
            
            print len(ctime_seg), len(ctime_seg[1])
            
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
                   
        nside=self._nside_out
        lmax=self._lmax
                
        sum_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
        
        mask = (freq>self.low_f) & (freq < self.high_f)
        freq = freq[mask]
        window = np.ones_like(freq)


        delf = self.fs/float(len(freq))#/len(strain[0]) #self.fs/4./len(strain[0]) SHOULD TAKE INTO ACCOUNT THE *2, THE NORMALISATION (1/L) AND THE DELTA F
        #geometry 
        
        npix_out = self.npix_out
        npix_in = self.npix_in
        
        mid_idx = int(len(ct_split)/2)
    
        vec_p_out = hp.pix2vec(self._nside_out,np.arange(npix_out))
        
        z_p = np.zeros(npix_out)
        A_pp = np.zeros((npix_out,npix_out))
        M_pp = np.zeros((npix_out,npix_out))
        
        for idx_b in range(self._nbase):
            
            print idx_b

            rot_m_array = self.rotation_pix(np.arange(npix_in), q_n[idx_b])  
            gammaI_rot = self.gammaI[idx_b][rot_m_array]
            gammaI_rot_ud = hp.ud_grade(gammaI_rot,nside_out = self._nside_out) 
            
            vec_b = hp.pix2vec(self._nside_in,pix_b[idx_b])
            bdotp = 2.*np.pi*np.dot(vec_b,vec_p_out)*self.R_earth/3.e8
            
            df = strains[idx_b]
            pf = pows[idx_b][mask]
            
            for ip in range(npix_out):
                z_p[ip] += 8.*np.pi/npix_out * delf*np.sum(window[:] 
                            * self.E_f(freq)[:]/ pf[:] * gammaI_rot_ud[ip]      ## minus sign? changed it to +
                            *(np.cos(bdotp[ip]*freq[:])*np.real(df[:]) + np.sin(bdotp[ip]*freq[:])*np.imag(df[:]))) 
                
                #A_pp[ip] +=  2.*(4.*np.pi)**2/npix_out**2 * delf**2 * np.sum(window[:]**2 * self.E_f(freq)[:]**2
                    * gammaI_rot_ud[ip] * gammaI_rot_ud[ip]*(np.cos((bdotp[ip]-bdotp[ip])*freq[:]) ))
                
                
                for jp in range(ip,npix_out):

                    val = 2.*(4.*np.pi)**2/npix_out**2 * delf**2 * np.sum(window[:]**2 * self.E_f(freq)[:]**2/ pf[:]
                    * gammaI_rot_ud[ip] * gammaI_rot_ud[jp]
                    *(np.cos((bdotp[ip]-bdotp[jp])*freq[:]) ))
                    M_pp[ip,jp] += val
                    
                    val2 = 2.*(4.*np.pi)**2/npix_out**2 * delf**2 * np.sum(window[:]**2 * self.E_f(freq)[:]**2
                    * gammaI_rot_ud[ip] * gammaI_rot_ud[jp]
                    *(np.cos((bdotp[ip]-bdotp[jp])*freq[:]) ))
                    A_pp[ip,jp] += val2
                    
                    if ip!= jp : 
                        M_pp[jp,ip] += val
                        A_pp[jp,ip] += val2
        
        
        #M_pp_inv = np.linalg.pinv(M_pp,rcond=1.e-5)
        
        #S_p = np.einsum('...ik,...k->...i', M_pp_inv, z_p)
        
        #fig = plt.figure()
        #hp.mollview(S_p)
        #plt.savefig('clean.pdf')
        if norm == False: return z_p, M_pp 
        else: return z_p, M_pp, A_pp #/norm


    def projector(self,ctime, s, psds, freqs,pix_bs, q_ns, norm = False):
        
        #just a summer wrapper really
        
        print 'proj run'
            
        nside=self._nside_out
        lmax=self._lmax
        
        pows = []
        
        for i in range(self._nbase):
            a, b = self.combo_tuples[i]
            pows.append(psds[a]*(psds[b]))      #(or sqrt)
                
        z_p, M_pp, A_pp = self.summer(ctime, s, pows, freqs, pix_bs, q_ns, norm)
        
        #print np.mean(data_lm)          
        
        if norm == False: return z_p, M_pp
        else: return z_p, M_pp, A_pp
        
    # ********* Scanner *********
    #to use scanner, re-check l, m ranges and other things. otherwise use scanner_1    

    def scan(self, ct_split, low_f, high_f, m_lm): 

        nside=self._nside_out
        lmax=self._lmax 
        npix = self.npix_out
        
        tstream = 0.+0.j
        t00 = 0.+0.j
        
        #        for idx_t, ct_split in enumerate(ctime):
            
        mid_idx = int(len(ct_split)/2)

        q_b = self.Q.azel2bore(np.degrees(self.az_b), np.degrees(self.el_b), None, None, np.degrees(self.H1.lon()), np.degrees(self.H1.lat()), ct_split[mid_idx])
        q_n = self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid), np.degrees(self.latMid), ct_split[mid_idx])[0]

        pix_b, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True) #spin-2
    
        p = pix_b
        print p          
        quat = q_n

        theta_b, phi_b = hp.pix2ang(nside,p)
        
        #print theta_b, phi_b
        
        rot_m_array = self.rotation_pix(np.arange(npix), quat) #rotating around the bisector of the gc 
        gammaI_rot = self.gammaI[rot_m_array]
        
        # Expand rotated gammas into lm
        glm = hp.map2alm(gammaI_rot, lmax, pol=False)              
        
        for l in range(lmax+1): #
            
            for m in range(-l,l+1): #
    
                idx_lm = hp.Alm.getidx(lmax,l,m)
                if m < 0: idx_lm = hp.Alm.getidx(lmax,l,-m)
                
                for lp in range(lmax+1):
                    
                    for mp in range(lp+1):
                        #print '+'
                        # remaining m index
                        mpp = m+mp
                        lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                        lmax_m = l + lp
                        
                        #print lp, mp, glm[hp.Alm.getidx(lmax,lp,mp)]
                        
                        for idxl, lpp in enumerate(range(lmin_m,lmax_m+1)):
                            tstream += ((-1)**mpp*self.freq_factor(lpp)*(0+1.j)**lpp
                            *(sph_harm(-mpp, lpp, theta_b, phi_b))
                                                *(glm[hp.Alm.getidx(lmax,lp,mp)])
                                                *np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*
                                                self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp]*m_lm[idx_lm]) 
                             
                    for mp in range(-lp,0):
                        
                        mpp = m+mp
                        lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                        lmax_m = l + lp
                        
                        #print lp, mp, ((-1)**mp*np.conj(glm[hp.Alm.getidx(lmax,lp,-mp)]))
                        
                        for idxl, lpp in enumerate(range(lmin_m,lmax_m+1)):
                            tstream += ((-1)**mpp*self.freq_factor(lpp,low_f,high_f)*(0+1.j)**lpp
                            *(sph_harm(-mpp, lpp, theta_b, phi_b))
                                                *((-1)**mp*np.conj(glm[hp.Alm.getidx(lmax,lp,-mp)]))
                                                *np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*
                                                self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp]*m_lm[idx_lm])
                            #print self.threej_m[lpp,l,lp,m,mp], 'negs'
                            
        return (tstream + np.conj(tstream))*np.ones_like(ct_split, dtype = complex)

    def M_lm_lpmp_t(self,ct_split, psds_split_t,freq,pix_b,q_n): 
        
        nside=self._nside_out
        lmax=self._lmax 
        al = hp.Alm.getidx(lmax,lmax,lmax)+1
            
        npix = self.npix_out       
        
        
        M_lm_lpmp = np.zeros((al,al), dtype = complex)
        #integral = np.ndarray(shape = (al,al), dtype = complex)
        #hit_lm = np.zeros(len(scanned_lm), dtype = complex)
        #print M_lm_lpmp
        
        mask = (freq>self.low_f) & (freq < self.high_f)
        freq = freq[mask]
        
        df = self.fs/float(len(freq))#/len(psds_split_t[0]) #self.fs/4./len(strain[0]) SHOULD TAKE INTO ACCOUNT THE *2, THE NORMALISATION (1/L) AND THE DELTA F
        print 'df: ', df
        #geometry
        
        mid_idx = int(len(ct_split)/2)
     
        # # get quaternions for H1->L1 baseline (use as boresight)
        # q_b = self.Q.azel2bore(np.degrees(self.az_b), np.degrees(self.el_b), None, None, np.degrees(self.H1.lon()), np.degrees(self.H1.lat()), ct_split[mid_idx])
        # # get quaternions for bisector pointing (use as boresight)
        # q_n = self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid), np.degrees(self.latMid), ct_split[mid_idx])[0]
        # pix_b, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True) #spin-2
        #
        # p = pix_b
        # quat = q_n
        # # polar angles of baseline vector
        theta_b, phi_b = hp.pix2ang(nside,pix_b)
        
        
        for idx_b in range(self._nbase):
            
            #print '==', idx_b, '++'
            
            a, b = self.combo_tuples[idx_b]
            weight = np.ones_like(psds_split_t[a])/(psds_split_t[a]*psds_split_t[b])
            weight = weight[mask]           #so we're only integrating on the interesting freqs
            
            rot_m_array = self.rotation_pix(np.arange(npix), q_n[idx_b]) #rotating around the bisector of the gc 
            gammaI_rot = self.gammaI[idx_b][rot_m_array]
            glm = hp.map2alm(gammaI_rot, lmax, pol=False)
            
            for l in range(lmax+1): #
                for m in range(l+1): #
                
                    idx_lm = hp.Alm.getidx(lmax,l,m)
                    #print idx_lm
                
                    for lt in range(lmax+1): #
                        for mt in range(lt+1): #
                
                            idx_ltmt = hp.Alm.getidx(lmax,lt,mt)
                            #print '(',idx_lm, idx_ltmt, ')'
                        
                            for lp in range(lmax+1): #
                                for mp in range(-lp,lp+1): #
                                    # remaining m index
                                    mpp = -(m+mp)
                                    lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                                    lmax_m = l + lp
                                
                                    for idxl, lpp in enumerate(range(lmin_m,lmax_m+1)):
                                        for lpt in range(lmax+1): #
                                            for mpt in range(-lpt,lpt+1): #
                                                # remaining mt index
                                                mppt = -(mt+mpt)
                                                lmin_mt = np.max([np.abs(lt - lpt), np.abs(mt + mpt)])
                                                lmax_mt = lt + lpt
                                            
                                                for idxlt, lppt in enumerate(range(lmin_mt,lmax_mt+1)):
                                                        
                                                    if mp > 0: rmp_rglm = glm[hp.Alm.getidx(lmax,lp,mp)] 
                                                    else: rmp_rglm = (-1)**mp*np.conj(glm[hp.Alm.getidx(lmax,lp,-mp)])
                                                    if mpt > 0:rmpt_rgltmt = glm[hp.Alm.getidx(lmax,lpt,mpt)] 
                                                    else: rmpt_rgltmt = (-1)**mpt*np.conj(glm[hp.Alm.getidx(lmax,lpt,-mpt)] )
                                                    
                                                    
                                                    M_lm_lpmp[idx_lm,idx_ltmt] += (np.sum(self.dfreq_factor(freq,lpp,idx_b)*
                                                    self.dfreq_factor(freq,lppt,idx_b)*weight)*df
                                                    *np.conj(4*np.pi*(0.+1.j)**lpp*np.conj(sph_harm(mpp, lpp, theta_b[idx_b], phi_b[idx_b]))
                                                    *rmp_rglm*self.coupK(lp,l,lpp,mp,m))*
                                                    (4*np.pi*(0.+1.j)**lppt*np.conj(sph_harm(mppt, lppt, theta_b[idx_b], phi_b[idx_b]))
                                                    *rmpt_rgltmt*self.coupK(lpt,lt,lppt,mpt,mt)))                                                    
                                                    
                                                    
                                                   
            #print M_lm_lpmp[idx_lm,idx_ltmt]                
        #norm = np.abs(np.sum(np.abs(hit_lm)))
        return M_lm_lpmp #hp.alm2map(scanned_lm/hits,nside,lmax=lmax)

    def M_lm_lpmp_t_2(self,ct_split, psds_split_t,freq,pix_b,q_n): 
        
        nside=self._nside_out
        lmax=self._lmax 
        al = hp.Alm.getidx(lmax,lmax,lmax)+1
            
        npix = self.npix_out       
        
        
        M_lm_lpmp = np.zeros((al,al), dtype = complex)
        #integral = np.ndarray(shape = (al,al), dtype = complex)
        #hit_lm = np.zeros(len(scanned_lm), dtype = complex)
        #print M_lm_lpmp
        
        mask = (freq>self.low_f) & (freq < self.high_f)
        freq = freq[mask]
        
        df = self.fs/float(len(freq))#/len(psds_split_t[0]) #self.fs/4./len(strain[0]) SHOULD TAKE INTO ACCOUNT THE *2, THE NORMALISATION (1/L) AND THE DELTA F
        print 'df: ', df
        #geometry
        
        mid_idx = int(len(ct_split)/2)
     
        # # get quaternions for H1->L1 baseline (use as boresight)
        # q_b = self.Q.azel2bore(np.degrees(self.az_b), np.degrees(self.el_b), None, None, np.degrees(self.H1.lon()), np.degrees(self.H1.lat()), ct_split[mid_idx])
        # # get quaternions for bisector pointing (use as boresight)
        # q_n = self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid), np.degrees(self.latMid), ct_split[mid_idx])[0]
        # pix_b, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True) #spin-2
        #
        # p = pix_b
        # quat = q_n
        # # polar angles of baseline vector
        theta_b, phi_b = hp.pix2ang(nside,pix_b)
        
        
        for idx_b in range(self._nbase):
            
            #print '==', idx_b, '++'
            
            a, b = self.combo_tuples[idx_b]
            weight = np.ones_like(psds_split_t[a])/(psds_split_t[a]*psds_split_t[b])
            weight = weight[mask]           #so we're only integrating on the interesting freqs
            
            rot_m_array = self.rotation_pix(np.arange(npix), q_n[idx_b]) #rotating around the bisector of the gc 
            gammaI_rot = self.gammaI[idx_b][rot_m_array]
            glm = hp.map2alm(gammaI_rot, lmax, pol=False)
            
            for l in range(lmax+1): #
                for m in range(l+1): #
                
                    idx_lm = hp.Alm.getidx(lmax,l,m)
                    #print idx_lm
                
                    for lt in range(lmax+1): #
                        for mt in range(lt+1): #
                
                            idx_ltmt = hp.Alm.getidx(lmax,lt,mt)
                            #print '(',idx_lm, idx_ltmt, ')'
                        
                            for lp in range(lmax+1): #
                                for mp in range(-lp,lp+1): #
                                    # remaining m index
                                    mpp = -(m+mp)
                                    lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                                    lmax_m = l + lp
                                
                                    for idxl, lpp in enumerate(range(lmin_m,lmax_m+1)):
                                        for lpt in range(lmax+1): #
                                            for mpt in range(-lpt,lpt+1): #
                                                # remaining mt index
                                                mppt = -(mt+mpt)
                                                lmin_mt = np.max([np.abs(lt - lpt), np.abs(mt + mpt)])
                                                lmax_mt = lt + lpt
                                            
                                                for idxlt, lppt in enumerate(range(lmin_mt,lmax_mt+1)):
                                                        
                                                    if mp > 0: rmp_rglm = glm[hp.Alm.getidx(lmax,lp,mp)] 
                                                    else: rmp_rglm = (-1)**mp*np.conj(glm[hp.Alm.getidx(lmax,lp,-mp)])
                                                    if mpt > 0:rmpt_rgltmt = glm[hp.Alm.getidx(lmax,lpt,mpt)] 
                                                    else: rmpt_rgltmt = (-1)**mpt*np.conj(glm[hp.Alm.getidx(lmax,lpt,-mpt)] )
                                                    
                                                    
                                                    M_lm_lpmp[idx_lm,idx_ltmt] += (np.sum(self.dfreq_factor(freq,lpp,idx_b)*
                                                    self.dfreq_factor(freq,lppt,idx_b)*weight)*df
                                                    *(4*np.pi*(0.+1.j)**lpp*np.conj(sph_harm(mpp, lpp, theta_b[idx_b], phi_b[idx_b]))
                                                    *rmp_rglm*self.coupK(lp,l,lpp,mp,m))*
                                                    (4*np.pi*(0.+1.j)**lppt*np.conj(sph_harm(mppt, lppt, theta_b[idx_b], phi_b[idx_b]))
                                                    *rmpt_rgltmt*self.coupK(lpt,lt,lppt,mpt,mt)))                                                    
                                                    
                                                    
                                                   
            #print M_lm_lpmp[idx_lm,idx_ltmt]                
        #norm = np.abs(np.sum(np.abs(hit_lm)))
        return M_lm_lpmp #hp.alm2map(scanned_lm/hits,nside,lmax=lmax)

    # ********* Decorrelator *********    

    # ********* S(f) *********

    #def S(self, s, freqs):
        
        #return 
