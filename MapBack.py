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

# LIGO-specific readligo.py 
import readligo as rl
import ligo_filter as lf
from gwpy.time import tconvert

'''

LIGO ANALYSIS ROUTINES

    *Fixed Setup Constants
    *Basic Tools
    *Data Segmenter
    *Projector
    *Scanner

'''

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
        
        else:
            dect_name = __import__(dect_name)
            #import name
            self._lon = dect_name.lon
            self._lat = dect_name.lat
            self._elev = dect_name.elev
            self._vec = dect_name.vec
        
        
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
        
        #hp.mollview(self.Fcross)
        #plt.savefig('Fc.pdf')
        
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
        
        return 1./norm * np.array([a,b,c])    
        
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

    def __init__(self, nside_in,nside_out, lmax, fs, low_f, high_f, dects, input_map = None): #Dect list
    
        self.Q = qp.QPoint(accuracy='low', fast_math=True, mean_aber=True)#, num_threads=1)
        
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
        self.npix_in = hp.nside2npix(self._nside_in)
        self.npix_out = hp.nside2npix(self._nside_out)

        # calculate overlap functions
        # TODO: integrate this with general detector table
        #theta, phi = hp.pix2ang(self._nside,np.arange(self.npix)) 
        
        self.gammaI = []
        
        for i in range(self._nbase):
            a, b = self.combo_tuples[i]
            self.gammaI.append((5./(8.*np.pi))*self.detectors[a].get_Fplus()*self.detectors[b].get_Fplus()+self.detectors[a].get_Fcross()*self.detectors[b].get_Fcross())
                
        hp.mollview(self.gammaI[0])
        plt.savefig('gammaI.pdf')

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

        self.hp = np.array(np.sqrt(np.abs(sfs.Istoke)/2))
        self.hc = np.array(np.sqrt(np.abs(sfs.Istoke)/2))

        #plt.figure()
        #hp.mollview(self.hp)
        #plt.loglog(freqs,np.sqrt(hf_psd(freqs)), color = 'g') #(freqs)
        #plt.ylim([-100.,100.])
        #plt.savefig('hp.png' )
                        
        
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
        
        plt.figure()
        plt.plot(win, color = 'r')
        plt.savefig('win.png' )
        
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
        nside = self._nside_out
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

    def buttering(self, x):
        nyq = 0.5*self.fs
        low = 30 / nyq
        high = 300 / nyq
        bb, ab = butter(4, [low, high], btype='band')
        bb = np.array(bb)
        ab = np.array(ab)
        butt_coefs = (bb,ab)
        num = 0.
        den = 0.
        for i in range(len(ab)):
            num += bb[i]*x**(-i)
            den += ab[i]*x**(-i)
        transfer = num/den
        
        return transfer

 #   def cutout(self,x, freqs,low = 20, high = 300):

    def injector(self,strains_in,ct_split,low_f,high_f,sim = False, simtyp = 'mono'):
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
        
        if sim == True:     #simulates streams for all detectors called when T.scope was initialised
            fakestreams = []
            q_xes = self.geometry_sim(ct_split)
            
            for (idx_det,dect) in enumerate(self.detectors):
                print idx_det
                q_x = q_xes[idx_det]
                fakestream = dect.simulate(freqs,q_x,simtyp) 
                fakestreams.append(fakestream)
                #plt.figure()
                #plt.plot(freqs,np.real(fakestream), c = 'red') 
                #plt.plot(freqs,np.imag(fakestream), c = 'blue') 
                #plt.savefig('fstreams.pdf' )
            #print np.std(fakestreams[0])
            #print np.std(fakestreams[1])
        
        ###
                
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
    
            Pxx, frexx = mlab.psd(strain_in_nowin, Fs=fs, NFFT=2*fs,noverlap=fs/2,window=np.blackman(2*fs),scale_by_freq=True)
            hf_psd = interp1d(frexx,Pxx)
            hf_psd_data = abs(hf_nowin.copy()*np.conj(hf_nowin.copy())/(fs**2))
    
    
            #Norm
            mask = (freqs>low_f) & (freqs < high_f)
            norm = np.mean(hf_psd_data[mask])/np.mean(hf_psd(freqs)[mask])
    
            #print norm
    
            hf_psd=interp1d(frexx,Pxx*norm)
            psds.append(hf_psd)
                
            #print frexx, Pxx, len(Pxx)
    
            #Pxx, frexx = mlab.psd(strain_in_win[:Nt], Fs = fs, NFFT = 4*fs, window = mlab.window_none)
            
            # plt.figure()
            # #plt.plot(freqs,) 
            # plt.savefig('.png' )

        if sim == True:
        
            if ndects == len(strains_in):

                for idx_det in range(len(strains_in)):
                    
                    rands = [np.random.normal(loc = 0., scale = 1. , size = len(hf_psd_data)),np.random.normal(loc = 0., scale = 1. , size = len(hf_psd_data))] 
                    fakenoise = rands[0]+1.j*rands[1]
                    fake_psd = psds[idx_det](freqs)*self.fs**2
                    fakenoise = np.array(fakenoise*np.sqrt(fake_psd/2.))#np.sqrt(self.fs/2.)#part of the normalization
        
        
                    fake = np.sum([fakenoise,fakestreams[idx_det]], axis=0)
        
                    fake_inv = np.fft.irfft(fake , n=2*Nt,norm = 'ortho')[:Nt]
                    # print 'fake[0]!'
                    # print fake[0]
                    # print 'average fakeinv!'
                    # print np.average(fake_inv)
                    # print 'std fakeinv'
                    # print np.std(fake_inv)
                    # print 'lens!'
                    # print len(fake), len(fake_inv)
                    faketot.append(fake_inv)
            else:
                
                for idx_det in range(ndects):
                    
                    rands = [np.random.normal(loc = 0., scale = 1. , size = len(hf_psd_data)),np.random.normal(loc = 0., scale = 1. , size = len(hf_psd_data))] 
                    fakenoise = rands[0]+1.j*rands[1]
                    fake_psd = psds[0](freqs)*self.fs**2            #SAME PSD FOR ALL DATA  
                    fakenoise = np.array(fakenoise*np.sqrt(fake_psd/2.))#np.sqrt(self.fs/2.)#part of the normalization
        
        
                    fake = np.sum([fakenoise,fakestreams[idx_det]], axis=0)
        
                    fake_inv = np.fft.irfft(fake , n=2*Nt,norm = 'ortho')[:Nt]
                    # print 'fake[0]!'
                    # print fake[0]
                    # print 'average fakeinv!'
                    # print np.average(fake_inv)
                    # print 'std fakeinv'
                    # print np.std(fake_inv)
                    # print 'lens!'
                    # print len(fake), len(fake_inv)
                    faketot.append(fake_inv)
        
        lenpsds = len(psds)            
        while ndects > lenpsds:
            psds.append(psds[0])
            lenpsds+=1
                                    
        
        return faketot, psds
        
        
        ####
        
   
    
    def filter(self,strain_in,low_f,high_f, hf_psd, simulate = False):
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
        
        notch_fs = np.array([14.0,34.70, 35.30, 35.90, 36.70, 37.30, 40.95, 60.00, 120.00, 179.99, 304.99, 331.49, 510.02, 1009.99])
        sigma_fs = np.array([.5,.5,.5,.5,.5,.5,.5,1.,1.,1.,1.,5.,5.,1.])
        #np.array([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.5,0.3,0.2])
        
        samp_hz = fs**2*(len(hf_copy))**(-1.)-6.68 #correction due to?
                
        pixels = np.arange(len(hf_copy))
             
        i = 0
          
        while i < len(notch_fs):
            notch_pix = int(notch_fs[i]*samp_hz)
            hf_nowin = hf_nowin*(1.-self.gaussian(pixels,notch_pix,sigma_fs[i]*samp_hz))
            i+=1           
        
        
        #BPING HF

        gauss_lo = self.halfgaussian(pixels,low_f*samp_hz,samp_hz)
        gauss_hi = self.halfgaussian(pixels,high_f*samp_hz,samp_hz)

        hf_nbped = hf_nowin*(1.-gauss_lo)*(gauss_hi)            ####
        
        #print 'average of band lim, std of band lim IN T DOMAIN'
        
        #hf_bp_inv= np.fft.irfft(hf_nbped, norm = 'ortho')
        
        #print np.average(hf_bp_inv), ' , ' , np.std(hf_bp_inv), '  ,  ', len(hf_bp_inv)
        #print np.average(strain_in_nowin), ' , ' , np.std(strain_in_nowin), '  ,  ', len(strain_in_nowin)
        #print np.average(strain_in_cp), ' , ' , np.std(strain_in_cp), '  ,  ', len(strain_in_cp)
        
        #plt.figure()
        #plt.plot(freqs,hf_nbped)
        #plt.savefig('show.png' )
        
        
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
        segs_H1 = rl.getsegs(start, stop, 'H1',flag='STOCH_CAT1', filelist=filelist)
        good_data_H1 = np.zeros(stop-start,dtype=np.bool)
        for (begin, end) in segs_H1:
            good_data_H1[begin-start:end-start] = True

        segs_L1 = rl.getsegs(start, stop, 'L1',flag='STOCH_CAT1', filelist=filelist)
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

    def summer(self, ct_split, strain, freq, pix_b, q_n ):     
                   
        nside=self._nside_out
        lmax=self._lmax
                
        sum_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
        m_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
        
        mask = (freq>self.low_f) & (freq < self.high_f)
        freq = freq[mask]


        df = self.fs/float(len(freq))#/len(strain[0]) #self.fs/4./len(strain[0]) SHOULD TAKE INTO ACCOUNT THE *2, THE NORMALISATION (1/L) AND THE DELTA F
        #geometry 
        
        npix = self.npix_out
        
        mid_idx = int(len(ct_split)/2)
    

        theta_b, phi_b = hp.pix2ang(nside,pix_b)
        
        
        for idx_b in range(self._nbase):
            
            rot_m_array = self.rotation_pix(np.arange(npix), q_n[idx_b])  
            gammaI_rot = self.gammaI[idx_b][rot_m_array]
            
            
            glm = hp.map2alm(gammaI_rot, lmax, pol=False)  
            
            # for ell in range(lmax+1):
            #     idxl0 =  hp.Alm.getidx(lmax,ell,0)
            #
            #     almbit = 0.
            #     for em in range(ell+1):
            #         idxlm =  hp.Alm.getidx(lmax,ell,em)
            #         almbit +=(2*glm[idxlm])*np.conj(glm[idxlm])/(2*ell+1)
            #
            #     print almbit - glm[idxl0]*np.conj(glm[idxl0])/(2*ell+1)
            
            # out :
            # (0.00779142015605+0j)
            # (1.40045923051e-35+0j)
            # (0.0013366737491+0j)
                        
            s = strain[idx_b][mask]
            
            for l in range(lmax+1):
                for m in range(l+1):

                    idx_lm = hp.Alm.getidx(lmax,l,m)

                    for lp in range(lmax+1):
                        for mp in range(-lp, lp+1):
                            # remaining m index
                            #Nmp = lp+1.
                            mpp = - (m+mp)
                            lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                            lmax_m = l + lp
                            for idxl, lpp in enumerate(range(lmin_m,lmax_m+1)):
                                if mp>0 :

                                    sum_lm[idx_lm] +=  (
                        
                                    4.*np.pi*(0+1.j)**lpp
                                    #*self.dfreq_factor(f,lpp)*s[idx_f]
                                    *np.conj(sph_harm(mpp, lpp, theta_b[idx_b], phi_b[idx_b]))*
                                    (
                                    glm[hp.Alm.getidx(lmax,lp,mp)]*self.coupK(lp,l,lpp,mp,m)
                                    ))*np.sum(self.dfreq_factor(freq,lpp,idx_b)*(s+np.conj(s)))*df    ##freq dependence summed over
                                    
                                    
                                else:
                                    sum_lm[idx_lm] += (
                        
                                     4.*np.pi*(0+1.j)**lpp
                                     *np.conj(sph_harm(mpp, lpp, theta_b[idx_b], phi_b[idx_b]))*
                                     (
                                     (-1.)**(mp)*np.conj(glm[hp.Alm.getidx(lmax,lp,-mp)])*self.coupK(lp,l,lpp,mp,m)
                                     ))*np.sum(self.dfreq_factor(freq,lpp,idx_b)*(s+np.conj(s)))*df
        sum_lm[0] = np.real(sum_lm[0])
        return sum_lm#/norm
 

    def projector(self,ctime, s_tuple,freqs,pix_bs, q_ns):
        
        #just a summer wrapper really
        
        print 'proj run'
            
        nside=self._nside_out
        lmax=self._lmax

        data_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
        s = []
            
        for i in range(self._nbase):
            a, b = self.combo_tuples[i]
            s.append(s_tuple[a]*np.conj(s_tuple[b]))
            
        data_lm += self.summer(ctime, s, freqs, pix_bs, q_ns)
        
        #print np.mean(data_lm)          
        
        return data_lm

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
                                                mppt = mt+mpt
                                                lmin_mt = np.max([np.abs(lt - lpt), np.abs(mt + mpt)])
                                                lmax_mt = lt + lpt
                                            
                                                for idxlt, lppt in enumerate(range(lmin_mt,lmax_mt+1)):
                                                        
                                                    if mp > 0:
                                                        
                                                        if mpt > 0:
                                                            
                                                            M_lm_lpmp[idx_lm,idx_ltmt] += (np.sum(self.dfreq_factor(freq,lpp,idx_b)*self.dfreq_factor(freq,lppt,idx_b)*weight)*df**2
                                                            *np.conj(4*np.pi*(0.+1.j)**lpp*np.conj(sph_harm(mpp, lpp, theta_b[idx_b], phi_b[idx_b]))
                                                            *glm[hp.Alm.getidx(lmax,lp,mp)]*self.coupK(lp,l,lpp,mp,m))*(4*np.pi*(0.+1.j)**lppt*np.conj(sph_harm(mppt, lppt, theta_b[idx_b], phi_b[idx_b]))
                                                            *glm[hp.Alm.getidx(lmax,lpt,mpt)]*self.coupK(lpt,lt,lppt,mpt,mt)))
                                                                                                                    
                                                        else:
                                                            
                                                            M_lm_lpmp[idx_lm,idx_ltmt] += (np.sum(self.dfreq_factor(freq,lpp,idx_b)*self.dfreq_factor(freq,lppt,idx_b)*weight)*df**2
                                                            *np.conj(4*np.pi*(0.+1.j)**lpp*np.conj(sph_harm(mpp, lpp, theta_b[idx_b], phi_b[idx_b]))
                                                            *glm[hp.Alm.getidx(lmax,lp,mp)]*self.coupK(lp,l,lpp,mp,m))*(4*np.pi*(0.+1.j)**lppt*np.conj(sph_harm(mppt, lppt, theta_b[idx_b], phi_b[idx_b]))
                                                            *(-1)**mpt*np.conj(glm[hp.Alm.getidx(lmax,lpt,-mpt)])*self.coupK(lpt,lt,lppt,mpt,mt)))
                                                                                                                
                                                    else:
                                                        
                                                        if mpt > 0:
                                                            
                                                            M_lm_lpmp[idx_lm,idx_ltmt] += (np.sum(self.dfreq_factor(freq,lpp,idx_b)*self.dfreq_factor(freq,lppt,idx_b)*weight)*df**2
                                                            *np.conj(4*np.pi*(0.+1.j)**lpp*np.conj(sph_harm(mpp, lpp, theta_b[idx_b], phi_b[idx_b]))
                                                            *(-1)**mp*np.conj(glm[hp.Alm.getidx(lmax,lp,-mp)])*self.coupK(lp,l,lpp,mp,m))*(4*np.pi*(0.+1.j)**lppt*np.conj(sph_harm(mppt, lppt, theta_b[idx_b], phi_b[idx_b]))
                                                            *glm[hp.Alm.getidx(lmax,lpt,mpt)]*self.coupK(lpt,lt,lppt,mpt,mt)))
                                                                                                                    
                                                        else:
                                                            
                                                            M_lm_lpmp[idx_lm,idx_ltmt] += (np.sum(self.dfreq_factor(freq,lpp,idx_b)*self.dfreq_factor(freq,lppt,idx_b)*weight)*df**2
                                                            *np.conj(4*np.pi*(0.+1.j)**lpp*np.conj(sph_harm(mpp, lpp, theta_b[idx_b], phi_b[idx_b]))
                                                            *(-1)**mp*np.conj(glm[hp.Alm.getidx(lmax,lp,-mp)])*self.coupK(lp,l,lpp,mp,m))*(4*np.pi*(0.+1.j)**lppt*np.conj(sph_harm(mppt, lppt, theta_b[idx_b], phi_b[idx_b]))
                                                            *(-1)**mpt*np.conj(glm[hp.Alm.getidx(lmax,lpt,-mpt)])*self.coupK(lpt,lt,lppt,mpt,mt)))
                                                                                                                    
                                                    
                                                    
                                                    
                                                   
            #print M_lm_lpmp[idx_lm,idx_ltmt]                
        
        #norm = np.abs(np.sum(np.abs(hit_lm)))
        return M_lm_lpmp #hp.alm2map(scanned_lm/hits,nside,lmax=lmax)
    
    def M_lm_lpmp_t_2(self,ct_split, psds_split_t,freq,pix_b,q_n): 
        
        nside=self._nside_out
        lmax=self._lmax 
        al = hp.Alm.getidx(lmax,lmax,lmax)+1
            
        npix = self.npix_out       
        
        
        M_lm_lpmp_2 = np.zeros((al,al), dtype = complex)
        #integral = np.ndarray(shape = (al,al), dtype = complex)
        #hit_lm = np.zeros(len(scanned_lm), dtype = complex)
        #print M_lm_lpmp
        
        mask = (freq>self.low_f) & (freq < self.high_f)
        freq = freq[mask]
        
        df = self.fs/float(len(freq))#/len(psds_split_t[0]) #self.fs/4./len(strain[0]) SHOULD TAKE INTO ACCOUNT THE *2, THE NORMALISATION (1/L) AND THE DELTA F
        
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
                                                mppt = mt+mpt
                                                lmin_mt = np.max([np.abs(lt - lpt), np.abs(mt + mpt)])
                                                lmax_mt = lt + lpt
                                            
                                                for idxlt, lppt in enumerate(range(lmin_mt,lmax_mt+1)):
                                                        
                                                    if mp > 0:
                                                        
                                                        if mpt > 0:
                                                            
                                                            M_lm_lpmp_2[idx_lm,idx_ltmt] += (np.sum(self.dfreq_factor(freq,lpp,idx_b)*self.dfreq_factor(freq,lppt,idx_b)*weight)*df**2
                                                            *(4*np.pi*(0.+1.j)**lpp*np.conj(sph_harm(mpp, lpp, theta_b[idx_b], phi_b[idx_b]))
                                                            *glm[hp.Alm.getidx(lmax,lp,mp)]*self.coupK(lp,l,lpp,mp,m))*(4*np.pi*(0.+1.j)**lppt*np.conj(sph_harm(mppt, lppt, theta_b[idx_b], phi_b[idx_b]))
                                                            *glm[hp.Alm.getidx(lmax,lpt,mpt)]*self.coupK(lpt,lt,lppt,mpt,mt)))
                                                            
                                                            if idx_lm == 3 & idx_ltmt ==3 : print M_lm_lpmp_2[idx_lm,idx_ltmt]
                                                        
                                                        else:
                                                            
                                                            M_lm_lpmp_2[idx_lm,idx_ltmt] += (np.sum(self.dfreq_factor(freq,lpp,idx_b)*self.dfreq_factor(freq,lppt,idx_b)*weight)*df**2
                                                            *(4*np.pi*(0.+1.j)**lpp*np.conj(sph_harm(mpp, lpp, theta_b[idx_b], phi_b[idx_b]))
                                                            *glm[hp.Alm.getidx(lmax,lp,mp)]*self.coupK(lp,l,lpp,mp,m))*(4*np.pi*(0.+1.j)**lppt*np.conj(sph_harm(mppt, lppt, theta_b[idx_b], phi_b[idx_b]))
                                                            *(-1)**mpt*np.conj(glm[hp.Alm.getidx(lmax,lpt,-mpt)])*self.coupK(lpt,lt,lppt,mpt,mt)))
                                                            
                                                            if idx_lm == 3 & idx_ltmt ==3 : print M_lm_lpmp_2[idx_lm,idx_ltmt]
                                                            
                                                    
                                                    else:
                                                        
                                                        if mpt > 0:
                                                            
                                                            M_lm_lpmp_2[idx_lm,idx_ltmt] += (np.sum(self.dfreq_factor(freq,lpp,idx_b)*self.dfreq_factor(freq,lppt,idx_b)*weight)*df**2
                                                            *(4*np.pi*(0.+1.j)**lpp*np.conj(sph_harm(mpp, lpp, theta_b[idx_b], phi_b[idx_b]))
                                                            *(-1)**mp*np.conj(glm[hp.Alm.getidx(lmax,lp,-mp)])*self.coupK(lp,l,lpp,mp,m))*(4*np.pi*(0.+1.j)**lppt*np.conj(sph_harm(mppt, lppt, theta_b[idx_b], phi_b[idx_b]))
                                                            *glm[hp.Alm.getidx(lmax,lpt,mpt)]*self.coupK(lpt,lt,lppt,mpt,mt)))
                                                            
                                                            if idx_lm == 3 & idx_ltmt ==3 : print M_lm_lpmp_2[idx_lm,idx_ltmt]
                                                            
                                                        
                                                        else:
                                                            
                                                            M_lm_lpmp_2[idx_lm,idx_ltmt] += (np.sum(self.dfreq_factor(freq,lpp,idx_b)*self.dfreq_factor(freq,lppt,idx_b)*weight)*df**2
                                                            *(4*np.pi*(0.+1.j)**lpp*np.conj(sph_harm(mpp, lpp, theta_b[idx_b], phi_b[idx_b]))
                                                            *(-1)**mp*np.conj(glm[hp.Alm.getidx(lmax,lp,-mp)])*self.coupK(lp,l,lpp,mp,m))*(4*np.pi*(0.+1.j)**lppt*np.conj(sph_harm(mppt, lppt, theta_b[idx_b], phi_b[idx_b]))
                                                            *(-1)**mpt*np.conj(glm[hp.Alm.getidx(lmax,lpt,-mpt)])*self.coupK(lpt,lt,lppt,mpt,mt)))
                                                            
                                                            if idx_lm == 3 & idx_ltmt ==3 : print M_lm_lpmp_2[idx_lm,idx_ltmt]
                                                            
                                                        
                                                    
                                                    
                                                    
                                                   
            #print M_lm_lpmp[idx_lm,idx_ltmt]                
        
        #norm = np.abs(np.sum(np.abs(hit_lm)))
        return M_lm_lpmp_2 #hp.alm2map(scanned_lm/hits,nside,lmax=lmax)
 

    # ********* Decorrelator *********    

    # ********* S(f) *********

    #def S(self, s, freqs):
        
        #return 
