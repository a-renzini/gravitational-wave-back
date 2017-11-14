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

class Ligo_Analyse(object):

    def __init__(self, nside, lmax, fs, low_f, high_f):
        self.Q = qp.QPoint(accuracy='low', fast_math=True, mean_aber=True)#, num_threads=1)
        
        self._nside = nside
        self._lmax = lmax
        self.fs = fs
        self.low_f = low_f
        self.high_f = high_f
        
        # ********* Fixed Setup Constants *********

        # Configuration: radians and metres, Earth-centered frame
        self.H1_lon = -2.08405676917
        self.H1_lat = 0.81079526383
        self.H1_elev = 142.554
        self.H1_vec = np.array([-2.16141492636e+06, -3.83469517889e+06, 4.60035022664e+06]) 

        self.L1_lon = -1.58430937078
        self.L1_lat = 0.53342313506
        self.L1_elev = -6.574
        self.L1_vec = np.array([-7.42760447238e+04, -5.49628371971e+06, 3.22425701744e+06])

        self.V1_lon = 0.18333805213
        self.V1_lat = 0.76151183984
        self.V1_elev = 51.884
        self.V1_vec = np.array([4.54637409900e+06, 8.42989697626e+05, 4.37857696241e+06])

        # work out viewing angle of baseline H1->L1
        self.az_b, self.el_b, self.baseline_length = self.vec2azel(self.H1_vec,self.L1_vec)
        # position of mid point and angle of great circle connecting to observatories
        self.latMid,self.lonMid, self.azMid = self.midpoint(self.H1_lat,self.H1_lon,self.L1_lat,self.L1_lon)
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
                        mpp = m+mp
                        lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                        lmax_m = l + lp
                        self.threej_m[lmin_m:lmax_m+1,l,lp,m,mp] = threej(l, lp, m, mp)
        
        # gamma functs
        self.npix = hp.nside2npix(self._nside)

        # calculate overlap functions
        # TODO: integrate this with general detector table
        theta, phi = hp.pix2ang(self._nside,np.arange(self.npix)) 
        self.gammaI = ofs.gammaIHL(theta,phi)
        self.gammaQ = ofs.gammaQHL(theta,phi)
        self.gammaU = ofs.gammaUHL(theta,phi)
        self.gammaV = ofs.gammaVHL(theta,phi)
        
        
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
        dec_quatmap,ra_quatmap = hp.pix2ang(self._nside,m_array) #
        quatmap = self.Q.radecpa2quat(np.rad2deg(ra_quatmap), np.rad2deg(dec_quatmap-np.pi*0.5), 0.*np.ones_like(ra_quatmap)) #but maybe orientation here is actually the orientation of detector a, b? in which case, one could input it as a variable!
        quatmap_rotated = np.ones_like(quatmap)
        i = 0
        while i < len(m_array): 
            quatmap_rotated[i] = qr.quat_mult(n,quatmap[i])
            i+=1
        quatmap_rot_pix = self.Q.quat2pix(quatmap_rotated,self._nside)[0] #rotated pixel list (polarizations are in [1])
        return quatmap_rot_pix

    def dfreq_factor(self,f,ell,alpha=3.,H0=68.0,f0=100.):
        # f : frequency (Hz)
        # ell : multipole
        # alpha: spectral index
        # b: baseline length (m)
        # f0: pivot frequency (Hz)
        # H0: Hubble rate today (km/s/Mpc)
        
        b=self.baseline_length
        
        km_mpc = 3.086e+19 # km/Mpc conversion
        c = 3.e8 # speed of light 
        #fac = 8.*np.pi**3/3./H0**2 * km_mpc**2 * f**3*(f/f0)**(alpha-3.) * spherical_jn(ell, 2.*np.pi*f*b/c)
        fac =  spherical_jn(ell, 2.*np.pi*(f)*b/c) #*(f/f0)**(alpha-3.) *
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
        nside = self._nside
        mid_idx = int(len(ct_split)/2)
    
        q_b = self.Q.azel2bore(np.degrees(self.az_b), np.degrees(self.el_b), None, None, np.degrees(self.H1_lon), np.degrees(self.H1_lat), ct_split[mid_idx])
        q_n = self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid), np.degrees(self.latMid), ct_split[mid_idx])[0]
    
        p, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True)
        n, s2n, c2n = self.Q.quat2pix(q_n, nside=nside, pol=True)  
        theta_b, phi_b = hp.pix2ang(nside,p)
        
        if pol == False: return p, q_n, n
        else : return p, s2p, c2p, q_n, n
            
# **************** Whitening Modules ***************

    def iir_bandstops(self, fstops, fs, order=4):
        """ellip notch filter
        fstops is a list of entries of the form [frequency (Hz), df, df2]                           
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

    def ligofilter(self, strain_in):
        NFFT = 4*self.fs
        Pxx, freqs = mlab.psd(strain_in, Fs = self.fs, NFFT = NFFT)
        psd = interp1d(freqs, Pxx)
        dt = 1./self.fs
        
        def white(strain, interp_psd, dt):
            Nt = len(strain)
            freqs = np.fft.rfftfreq(Nt, dt)

            # whitening: transform to freq domain, divide by asd, then transform back, 
            # taking care to get normalization right.
            hf = np.fft.rfft(strain)
            norm = 1./np.sqrt(1./(dt*2))
            white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
            white_ht = np.fft.irfft(white_hf, n=Nt)
            plt.figure()
            plt.plot(freqs,abs(hf), color = 'b')
            plt.plot(freqs,np.sqrt(interp_psd(freqs)), color = 'r')
            #plt.plot(hf2_inv, color = 'r')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim([20.,1000.])
            #plt.ylim([-5,5])
            plt.savefig('test1.png' )
            return white_ht

        # now whiten the data from H1 and L1, and the template (use H1 PSD):
        strain_white = white(strain_in,psd,dt)
    
        # We need to suppress the high frequency noise (no signal!) with some bandpassing:
        #fband = [43.0, 300.0]
        
        #bb, ab = butter(4, [fband[0]*2./self.fs, fband[1]*2./self.fs], btype='band')
        #normalization = np.sqrt((fband[1]-fband[0])/(self.fs/2))
        #strain_whitebp = filtfilt(bb, ab, strain_white) / normalization
        
        plt.figure()
        plt.plot(strain_white, color = 'b')
        #plt.plot(hf2_inv, color = 'r')
        #plt.xscale('log')
        #plt.yscale('log')
        #plt.xlim([20.,1000.])
        plt.ylim([-5,5])
        plt.savefig('testnbp.png' )

    def injector(self,strain_in,low_f,high_f):
        fs=self.fs        
        dt=1./fs
        
        '''WINDOWING & RFFTING.'''
        
        Nt = len(strain_in)
        Nt = lf.bestFFTlength(Nt)
        strain_in = strain_in[:Nt]
        strain_in_nowin = np.copy(strain_in)
        strain_in_nowin *= signal.tukey(Nt,alpha=0.05)
        strain_in *= np.blackman(Nt)
        freqs = np.fft.rfftfreq(2*Nt, dt)
        hf = np.fft.rfft(strain_in, n=2*Nt)#, norm = 'ortho') 
        hf_nowin = np.fft.rfft(strain_in_nowin, n=2*Nt)#, norm = 'ortho') 
        
        hf = hf[:Nt/2+1]
        hf_nowin = hf_nowin[:Nt/2+1]
        freqs = freqs[:Nt/2+1]
                
        '''the PSD. '''
        
        Pxx, frexx = mlab.psd(strain_in_nowin, Fs=fs, NFFT=2*fs,noverlap=fs/2,window=np.blackman(2*fs),scale_by_freq=True)
        hf_psd = interp1d(frexx,Pxx)
        hf_psd_data = abs(hf_nowin.copy()*np.conj(hf_nowin.copy())/(fs**2))
        
        
        #Norm
        mask = (freqs>low_f) & (freqs < high_f)
        norm = np.mean(hf_psd_data[mask])/np.mean(hf_psd(freqs)[mask])
        
        #print norm
        
        hf_psd=interp1d(frexx,Pxx*norm)
        
        #print frexx, Pxx, len(Pxx)
        
        #Pxx, frexx = mlab.psd(strain_in_win[:Nt], Fs = fs, NFFT = 4*fs, window = mlab.window_none)
                
        # plt.figure()
        # plt.loglog(freqs,np.sqrt(hf_psd_data), color = 'r')
        # plt.loglog(frexx,np.sqrt(Pxx), alpha = 0.5)
        # #plt.loglog(freqs,np.sqrt(hf_psd(freqs)), color = 'g') #(freqs)
        # #plt.ylim([-100.,100.])
        # plt.savefig('pxx.png' )

        print '===='
        rands = [np.random.normal(loc = 0., scale = 1. , size = len(hf_psd_data)),np.random.normal(loc = 0., scale = 1. , size = len(hf_psd_data))] 
        print '++++'
        fake = rands[0]+1.j*rands[1]

        fake_psd = hf_psd(freqs)*self.fs**2
        fake = fake*np.sqrt(fake_psd/2.)#np.sqrt(self.fs/2.)#part of the normalization
        
        fakeinv = np.fft.irfft(fake, n=2*Nt) 
        # plt.figure()
        # plt.loglog(freqs,abs(fake), color = 'r')
        # plt.loglog(freqs,np.sqrt(fake_psd),color = 'b')
        # plt.xlim(20.,1000.)
        # plt.savefig('fakeinv.png' )
        #
        # fake = fake#*np.sqrt(Nt)
        #
        #
        # plt.figure()
        # plt.plot(fakeinv, color = 'r')
        # plt.savefig('fakeinv1.png' )
        #
        # fakeagain = np.fft.rfft(fakeinv, norm = 'ortho')
        
        return fakeinv[:Nt]
    
    def filter(self,strain_in,low_f,high_f):
        fs=self.fs        
        dt=1./fs
        
        '''WINDOWING & RFFTING.'''
        
        Nt = len(strain_in)
        Nt = lf.bestFFTlength(Nt)
        strain_in = strain_in[:Nt]
        strain_in_nowin = np.copy(strain_in)
        strain_in_nowin *= signal.tukey(Nt,alpha=0.05)
        strain_in *= np.blackman(Nt)
        freqs = np.fft.rfftfreq(2*Nt, dt)
        #print '=rfft='
        hf = np.fft.rfft(strain_in, n=2*Nt)#, norm = 'ortho') 
        hf_nowin = np.fft.rfft(strain_in_nowin, n=2*Nt)#, norm = 'ortho') 
        #print '++'
        
        hf = hf[:Nt/2+1]
        hf_nowin = hf_nowin[:Nt/2+1]
        freqs = freqs[:Nt/2+1]
                
        '''the PSD. '''
        Pxx, frexx = mlab.psd(strain_in_nowin, Fs=fs, NFFT=2*fs,noverlap=fs/2,window=np.blackman(2*fs),scale_by_freq=True)
        hf_psd = interp1d(frexx,Pxx)
        hf_psd_data = abs(hf_nowin.copy()*np.conj(hf_nowin.copy())/(fs**2))
        
        #Norm
        mask = (freqs>low_f) & (freqs < high_f)
        norm = np.mean(hf_psd_data[mask])/np.mean(hf_psd(freqs)[mask])
        
        #print norm
        
        hf_psd=interp1d(frexx,Pxx*norm)
        
        
        '''NOTCHING. '''
        
        hf_in = hf.copy()
        notch_fs = np.array([14.0,34.70, 35.30, 35.90, 36.70, 37.30, 40.95, 60.00, 120.00, 179.99, 304.99, 331.49, 510.02, 1009.99])
        sigma_fs = np.array([.5,.5,.5,.5,.5,.5,.5,1.,1.,1.,1.,5.,5.,1.])
        #np.array([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.5,0.3,0.2])
        
        samp_hz = fs**2*(len(hf))**(-1.)-6.68 #correction due to?
                
        pixels = np.arange(len(hf))

        #hf_psd = abs(hf_win.copy())**2
        #hf_psd_in = hf_psd.copy()
             
        i = 0
          
        while i < len(notch_fs):
            notch_pix = int(notch_fs[i]*samp_hz)
            hf_nowin = hf_nowin*(1.-self.gaussian(pixels,notch_pix,sigma_fs[i]*samp_hz))
            i+=1           
        
        #FITTING PSD. ffit(self,f,a,b,c,d,e)
        
        #cropping:
        #mask = np.ones(len(freqs),dtype=bool)
            
        #for (j,notch) in enumerate(notch_fs):      
        #    for (i,f) in enumerate(freqs):
        #        if abs(f-notch) < 1.: mask[i] = False
        
        #low_f = 50.
        #high_f = 350.
        
        # for (f,hfi,bo) in zip(freqs[low_f*int(samp_hz):high_f*int(samp_hz)],hf_psd[low_f*int(samp_hz):high_f*int(samp_hz)],mask[low_f*int(samp_hz):high_f*int(samp_hz)]):
        #      if bo == True:
        #          freqscut.append(f)
        #          hf_psdcut.append(hfi)

        #psd_params, psd_cov = curve_fit(self.ffit2,freqscut,hf_psdcut,p0=(40.,200.,3.21777e-44))
        
        #fitted = np.array(self.ffit2(freqs[1:],psd_params[0],psd_params[1],psd_params[2]))
        
        #BPING HF
        
        gauss_lo = self.halfgaussian(pixels,low_f*samp_hz,samp_hz)
        gauss_hi = self.halfgaussian(pixels,high_f*samp_hz,samp_hz)
        hf_nbped = hf_nowin*(1.-gauss_lo)*(gauss_hi)
              
        
        # whitening: transform to freq domain, divide by asd
        # remember: interp_psd is strain/rtHz
        # white_hf = hf_nbped/(np.sqrt(hf_psd(freqs)/2./dt))#hf_psd_in[1:]
        # white_hf[0] = 0.
        # white_hf[-1:] = 0.
        #white_hf_bp = white_hf*self.g_butt(freqs,3)
        
        #hf_inv = np.fft.irfft(white_hf, n=Nt)
        
        # index = [idx, dect]
        #
            
        return hf_nbped, hf_psd
        

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
       
    def summer(self, ct_split, s, freq, pix_b, q_n):     
           
        nside=self._nside
        lmax=self._lmax
        
        df = self.fs/4./len(s)

        sum_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
        hit_lm = np.zeros_like(sum_lm)
        
        mask = (freq>self.low_f) & (freq < self.high_f)
        freq = freq[mask]
        s = s[mask]

        #geometry 
        
        npix = self.npix
        
        mid_idx = int(len(ct_split)/2)
    
        # # get quaternions for H1->L1 baseline (use as boresight)
        # q_b = self.Q.azel2bore(np.degrees(self.az_b), np.degrees(self.el_b), None, None, np.degrees(self.H1_lon), np.degrees(self.H1_lat), ct_split[mid_idx])
        # # get quaternions for bisector pointing (use as boresight)
        # q_n = self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid), np.degrees(self.latMid), ct_split[mid_idx])[0]
        # pix_b, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True) #spin-2
        #
        # p = pix_b
        # quat = q_n
        # # polar angles of baseline vector
        theta_b, phi_b = hp.pix2ang(nside,pix_b)
                
        # rotate gammas
        # TODO: will need to oversample here
        # i.e. use nside > nside final map
        # TODO: pol gammas
        rot_m_array = self.rotation_pix(np.arange(npix), q_n) #rotating around the bisector of the gc 
        gammaI_rot = self.gammaI[rot_m_array]

        # Expand rotated gammas into lm
        glm = hp.map2alm(gammaI_rot, lmax, pol=False)              
        
        '''
        
        insert integral over frequency in the sum; integrate.quad(self.dfreq_factor*s,fmin,fmax,args=(ell))[0]
        s needs to be turned into a function of frequency(?)
        
        '''

        # sum over lp, mp
        for l in range(lmax+1):
            for m in range(l+1):

                idx_lm = hp.Alm.getidx(lmax,l,m)

                for lp in range(lmax+1):
                    for mp in range(lp+1):
                        # remaining m index
                        mpp = m+mp
                        lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                        lmax_m = l + lp
                        for idxl, lpp in enumerate(range(lmin_m,lmax_m+1)):

                            sum_lm[idx_lm] += np.conj(4*np.pi*(0+1.j)**lpp
                            #*self.dfreq_factor(f,lpp)*s[idx_f]
                            *np.conj(sph_harm(mpp, lpp, theta_b, phi_b))*
                            glm[hp.Alm.getidx(lmax,lp,mp)]*np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*
                            self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp])*np.sum(self.dfreq_factor(freq,lpp)*s)*df    ##freq dependence summed over
                                                
                            #hit_lm[idx_lm] += np.conj((-1)**mpp*(0+1.j)**lpp*np.sum(self.dfreq_factor(freq,lpp))*df*np.conj(sph_harm(mpp, lpp, theta_b, phi_b))*glm[hp.Alm.getidx(lmax,lp,mp)]*np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp])

        #norm = np.abs(np.sum(np.abs(hit_lm)))
        return sum_lm#/norm


    def summer_f(self, ct_split, f):        #returns summed element for a specific f
        nside=self._nside
        lmax=self._lmax
        
        sum_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
                
        npix = self.npix
        mid_idx = int(len(ct_split)/2)
    
        q_b = self.Q.azel2bore(np.degrees(self.az_b), np.degrees(self.el_b), None, None, np.degrees(self.H1_lon), np.degrees(self.H1_lat), ct_split[mid_idx])
        q_n = self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid), np.degrees(self.latMid), ct_split[mid_idx])[0]

        pix_b, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True) #spin-2
        
        p = pix_b          
        quat = q_n
    
        theta_b, phi_b = hp.pix2ang(nside,p)
        
        rot_m_array = self.rotation_pix(np.arange(npix), quat) #rotating around the bisector of the gc 
        gammaI_rot = self.gammaI[rot_m_array]

        glm = hp.map2alm(gammaI_rot, lmax, pol=False)              
        
        hits = 0.
        # sum over lp, mp
        for l in range(lmax+1):
            for m in range(l+1):
                idx_lm = hp.Alm.getidx(lmax,l,m)
                #print idx_f
                for lp in range(lmax+1):
                    for mp in range(lp+1):
                        # remaining m index
                        mpp = m+mp
                        lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                        lmax_m = l + lp
                        for idxl, lpp in enumerate(range(lmin_m,lmax_m+1)):
                            sum_lm[idx_lm] += ((-1)**mpp*(0+1.j)**lpp*self.dfreq_factor(f,lpp)
                            *sph_harm(mpp, lpp, theta_b, phi_b)*
                                                glm[hp.Alm.getidx(lmax,lp,mp)]*np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*
                                                self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp]) #sure about these 3js?
                                                    #s[idx_f]
                            hits += 1.
        return sum_lm/hits

    '''
    def summer_f_lm(self, ct_split, f,idx_lm):        
        nside=self._nside
        lmax=self._lmax
        
        sum_lm = 0.
                
        npix = self.npix
        mid_idx = int(len(ct_split)/2)
    
        q_b = self.Q.azel2bore(np.degrees(self.az_b), np.degrees(self.el_b), None, None, np.degrees(self.H1_lon), np.degrees(self.H1_lat), ct_split[mid_idx])
        q_n = self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid), np.degrees(self.latMid), ct_split[mid_idx])[0]

        pix_b, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True) #spin-2
        
        p = pix_b          
        quat = q_n
    
        theta_b, phi_b = hp.pix2ang(nside,p)
        
        rot_m_array = self.rotation_pix(np.arange(npix), quat) #rotating around the bisector of the gc 
        gammaI_rot = self.gammaI[rot_m_array]

        glm = hp.map2alm(gammaI_rot, lmax, pol=False)              
        
        hits = 0.
        
        l, m = hp.Alm.getlm(lmax, i=idx_lm)
        
        for lp in range(lmax+1):
            for mp in range(lp+1):
                # remaining m index
                mpp = m+mp
                lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                lmax_m = l + lp
                for idxl, lpp in enumerate(range(lmin_m,lmax_m+1)):
                    sum_lm += ((-1)**mpp*(0+1.j)**lpp*self.dfreq_factor(f,lpp)
                    *sph_harm(mpp, lpp, theta_b, phi_b)*
                                        glm[hp.Alm.getidx(lmax,lp,mp)]*np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*
                                        self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp]) #sure about these 3js?
                                            #s[idx_f]
                    hits += 1.
        return sum_lm/hits
    '''
        
    def projector(self,ctime, strain_H1, strain_H2,freqs,pix_bs, q_ns):
        
        print 'proj run'    
        nside=self._nside
        lmax=self._lmax

        data_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
        #hits = 0.
        for idx_t, (ct_split, s_1, s_2, pix_b, q_n) in enumerate(zip(ctime, strain_H1, strain_H2,pix_bs, q_ns)): 
                                    
            s = s_1*np.conj(s_2) ##  still fully frequency dependent
            data_lm += self.summer(ct_split, s, freqs, pix_b, q_n)
                        
            #hits+=1.
        
        return data_lm#/hits

    # ********* Scanner *********
    #to use scanner, re-check l, m ranges and other things. otherwise use scanner_1    

    def scan(self, ct_split, low_f, high_f, m_lm): 

        nside=self._nside
        lmax=self._lmax 
        npix = self.npix
        
        tstream = 0.+0.j
        t00 = 0.+0.j
        
        #        for idx_t, ct_split in enumerate(ctime):
            
        mid_idx = int(len(ct_split)/2)

        q_b = self.Q.azel2bore(np.degrees(self.az_b), np.degrees(self.el_b), None, None, np.degrees(self.H1_lon), np.degrees(self.H1_lat), ct_split[mid_idx])
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

    def M_lm_lpmp_t(self,ct_split, p_split_1_t, p_split_2_t,freq): 
        
        nside=self._nside
        lmax=self._lmax 
        al = hp.Alm.getidx(lmax,lmax,lmax)+1
            
        npix = self.npix
        
        df = self.fs/4./len(p_split_1_t)
        weight = 1./(p_split_1_t*p_split_2_t) 
        
        M_lm_lpmp = np.zeros((al,al), dtype = complex)
        #integral = np.ndarray(shape = (al,al), dtype = complex)
        #hit_lm = np.zeros(len(scanned_lm), dtype = complex)
        #print M_lm_lpmp
        
        mask = (freq>self.low_f) & (freq < self.high_f)
        freq = freq[mask]
        weight = weight[mask]
        
        #geometry
        mid_idx = int(len(ct_split)/2)
        
        q_b = self.Q.azel2bore(np.degrees(self.az_b), np.degrees(self.el_b), None, None, np.degrees(self.H1_lon), np.degrees(self.H1_lat), ct_split[mid_idx])
        q_n = self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid), np.degrees(self.latMid), ct_split[mid_idx])[0]
        
        p, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True) 
        quat = q_n
        theta_b, phi_b = hp.pix2ang(nside,p)
        
        rot_m_array = self.rotation_pix(np.arange(npix), quat) 
        gammaI_rot = self.gammaI[rot_m_array]
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
                            for mp in range(lp+1): #
                                # remaining m index
                                mpp = m+mp
                                lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                                lmax_m = l + lp
                                
                                for idxl, lpp in enumerate(range(lmin_m,lmax_m+1)):
                                    for lpt in range(lmax+1): #
                                        for mpt in range(lpt+1): #
                                            # remaining mt index
                                            mppt = mt+mpt
                                            lmin_mt = np.max([np.abs(lt - lpt), np.abs(mt + mpt)])
                                            lmax_mt = lt + lpt
                                            
                                            for idxlt, lppt in enumerate(range(lmin_mt,lmax_mt+1)):
                                                #integral[idx_lm,idx_ltmt] = np.sum(self.dfreq_factor(freq,lpp)*self.dfreq_factor(freq,lppt)*weight)*df**2
                                                M_lm_lpmp[idx_lm,idx_ltmt] += np.sum(self.dfreq_factor(freq,lpp)*self.dfreq_factor(freq,lppt)*weight)*df**2
                                                np.conj(4*np.pi*(0+1.j)**lpp*np.conj(sph_harm(mpp, lpp, theta_b, phi_b))
                                                *glm[hp.Alm.getidx(lmax,lp,mp)]*np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp])*(4*np.pi*(0+1.j)**lppt*np.conj(sph_harm(mppt, lppt, theta_b, phi_b))
                                                *glm[hp.Alm.getidx(lmax,lpt,mpt)]*np.sqrt((2*lt+1)*(2*lpt+1)*(2*lppt+1)/4./np.pi)*self.threej_0[lppt,lt,lpt]*self.threej_m[lppt,lt,lpt,mt,mpt])
            
            #print M_lm_lpmp[idx_lm,idx_ltmt]                
                                    #hit_lm[idx_lm] += ((-1)**mpp*(0+1.j)**lpp*np.sum(self.dfreq_factor(freq,lpp))*df*sph_harm(mpp, lpp, theta_b,phi_b)*glm[hp.Alm.getidx(lmax,lp,mp)]*np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp])
        
        #norm = np.abs(np.sum(np.abs(hit_lm)))
        return M_lm_lpmp #hp.alm2map(scanned_lm/hits,nside,lmax=lmax)
    

    def scanner(self,ct_split, p_split_1_t, p_split_2_t,freq,data_lm = 1.): 
        
        nside=self._nside
        lmax=self._lmax 
            
        npix = self.npix
        
        df = self.fs/4./len(p_split_1_t)
        
        scanned_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
        hit_lm = np.zeros(len(scanned_lm), dtype = complex)
        
        #geometry
        mid_idx = int(len(ct_split)/2)
        
        q_b = self.Q.azel2bore(np.degrees(self.az_b), np.degrees(self.el_b), None, None, np.degrees(self.H1_lon), np.degrees(self.H1_lat), ct_split[mid_idx])
        q_n = self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid), np.degrees(self.latMid), ct_split[mid_idx])[0]
        
        p, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True) 
        quat = q_n
        theta_b, phi_b = hp.pix2ang(nside,p)
        
        rot_m_array = self.rotation_pix(np.arange(npix), quat) 
        gammaI_rot = self.gammaI[rot_m_array]
        glm = hp.map2alm(gammaI_rot, lmax, pol=False)

        weight = 1./(p_split_1_t*p_split_2_t) 
        
        for l in range(lmax+1): #
            for m in range(l+1): #
                
                idx_lm = hp.Alm.getidx(lmax,l,m)

                for lp in range(lmax+1): #
                    for mp in range(lp+1): #
                        # remaining m index
                        mpp = m+mp
                        lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                        lmax_m = l + lp
                        for idxl, lpp in enumerate(range(lmin_m,lmax_m+1)):

                            scanned_lm[idx_lm] += np.conj(4*np.pi*(0+1.j)**lpp*np.conj(sph_harm(mpp, lpp, theta_b, phi_b))
                            *glm[hp.Alm.getidx(lmax,lp,mp)]*np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp])*np.sum(self.dfreq_factor(freq,lpp)*weight*data_lm)*df
                            
                            hit_lm[idx_lm] += ((-1)**mpp*(0+1.j)**lpp*np.sum(self.dfreq_factor(freq,lpp))*df
                            *sph_harm(mpp, lpp, theta_b, phi_b)*
                                                glm[hp.Alm.getidx(lmax,lp,mp)]*np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*
                                                self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp])
        
        norm = np.abs(np.sum(np.abs(hit_lm)))
        return scanned_lm/norm #hp.alm2map(scanned_lm/hits,nside,lmax=lmax)
        
    def scanner_short(self,ctime_idx_t, idx_f, p_split_1_f, p_split_2,freq,data_lm = 1.): #scanner(ctime_array, idx_t, p_split_1[idx_t], p_split_2, freq_coar_array)
    
        nside=self._nside
        lmax=self._lmax 
            
        npix = self.npix
        
        scanned_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
        hit_lm = np.zeros(len(scanned_lm))
        hits = 0.
        
        mid_idx = int(len(ctime_idx_t)/2)
        
        q_b = self.Q.azel2bore(np.degrees(self.az_b), np.degrees(self.el_b), None, None, np.degrees(self.H1_lon), np.degrees(self.H1_lat), ctime_idx_t[mid_idx])
        q_n = self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid), np.degrees(self.latMid), ctime_idx_t[mid_idx])[0]
        
        p, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True) 
        
        quat = q_n
        
        theta_b, phi_b = hp.pix2ang(nside,p)
        
        rot_m_array = self.rotation_pix(np.arange(npix), quat) 
        gammaI_rot = self.gammaI[rot_m_array]

        glm = hp.map2alm(gammaI_rot, lmax, pol=False)
        
        for idx_fp, fp in enumerate(freq):
            weight = np.divide(1.,p_split_1_f*p_split_2[idx_fp])
            for l in range(lmax+1): #
                for m in range(l+1): #
                    #print l, m
                    idx_lm = hp.Alm.getidx(lmax,l,m)
                    for lp in range(lmax+1): #
                        for mp in range(lp+1): #
                            # remaining m index
                            mpp = m+mp
                            lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                            lmax_m = l + lp
                            for idxl, lpp in enumerate(range(lmin_m,lmax_m+1)):
                                scanned_lm[idx_lm] += ((-1)**mpp*(0+1.j)**lpp*sph_harm(mpp, lpp, theta_b, phi_b)*self.dfreq_factor(fp,lpp)
                                *glm[hp.Alm.getidx(lmax,lp,mp)]*np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp])*data_lm*weight
                                hit_lm[idx_lm] += 1.
                                hits += 1.
        return scanned_lm/hits #hp.alm2map(scanned_lm/hits,nside,lmax=lmax)
    
    # ********* Decorrelator *********    

  
    def decorrelator(self, ctime, freq, p_split_1, p_split_2):
        
        nside=self._nside
        lmax=self._lmax 
            
        npix = self.npix        
        
        M_lm_lpmp =[]
        
        for idx_lm in range(hp.Alm.getidx(lmax,lmax,lmax)+1):
            print idx_lm
            M_lpmp = np.array([np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)])
                
            for idx_t, ct_split in enumerate(ctime):
                print idx_t
                for idx_f, f in enumerate(freq[idx_t]):
                    print '+'
                    scan_lm = self.scanner_short(ct_split, idx_f,p_split_1[idx_t][idx_f], p_split_2[idx_t],freq[idx_t])
                    print '-'
                    M_lpmp += self.summer_f_lm(ct_split, f,idx_lm)*scan_lm
                
            M_lm_lpmp.append(M_lpmp)
        return M_lm_lpmp
    
    # ********* S(f) *********

    #def S(self, s, freqs):
        
        #return 