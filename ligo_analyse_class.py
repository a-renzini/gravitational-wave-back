import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.special import spherical_jn, sph_harm
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
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

    def __init__(self, nside, lmax, fs):
        self.Q = qp.QPoint(accuracy='low', fast_math=True, mean_aber=True)#, num_threads=1)
        
        self._nside = nside
        self._lmax = lmax
        self.fs = fs
        
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
            for m in range(l+1):
                for lp in range(lmax+1):
                    lmin0 = np.abs(l - lp)
                    lmax0 = l + lp
                    self.threej_0[lmin0:lmax0+1,l,lp] = threej(l, lp, 0, 0)
                    for mp in range(lp+1):
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
        fac =  spherical_jn(ell, 2.*np.pi*(f+0.j)*b/c) #*(f/f0)**(alpha-3.) *
        # add band pass and notches here
 
        return fac

    def freq_factor(self,ell,fmin,fmax,alpha=3.,H0=68.0,f0=100.):
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
    
    def get_filter_coefs(self, fs, bandpass=True):
    
        # assemble the filter b,a coefficients:
        coefs = []

        # bandpass filter parameters
        lowcut=20 #43
        highcut=300 #260
        order = 4

        # Frequencies of notches at known instrumental spectral line frequencies.
        # You can see these lines in the ASD above, so it is straightforward to make this list.
        notchesAbsolute = np.array(
            [14.0,34.70, 35.30, 35.90, 36.70, 37.30, 40.95, 60.00, 
             120.00, 179.99, 304.99, 331.49, 510.02, 1009.99])
        # exclude notch below lowcut
        notchesAbsolute = notchesAbsolute[notchesAbsolute > lowcut]

        # notch filter coefficients:
        for notchf in notchesAbsolute:                      
            bn, an = self.iir_bandstops(np.array([[notchf,1,0.1]]), fs, order=4)
            coefs.append((bn,an))

        # Manually do a wider notch filter around 510 Hz etc.          
        bn, an = self.iir_bandstops(np.array([[510,200,20]]), fs, order=4)
        coefs.append((bn, an))

        # also notch out the forest of lines around 331.5 Hz
        bn, an = self.iir_bandstops(np.array([[331.5,10,1]]), fs, order=4)
        coefs.append((bn, an))

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
        


    def whiten(self, strain, interp_psd, dt):
        Nt = len(strain)
        Nt = lf.bestFFTlength(Nt)
        freqs = np.fft.rfftfreq(Nt, dt)
        print 'whitening...'
        # whitening: transform to freq domain, divide by asd
        hf = np.fft.rfft(strain[:Nt])/np.sqrt(Nt) #?
        # remember: interp_psd is strain/rtHz
        white_hf = hf/(np.sqrt(interp_psd(freqs)/2./dt))
	    #white_hf_bp = white_hf*self.g_butt(freqs,3)
        print 'done whitening...'
        return white_hf

    def filter(self, strain_in):
        #print strain_in
        fs=self.fs
        dt=1./fs
        strain = strain_in.copy()
        # do bandpass and notch filtering
        # get filter coefficients
        coefs = self.get_filter_coefs(fs,bandpass=False)
        
        # filter it:
        strain_fil = self.filter_data(strain,coefs)
        
        # bandpass filter parameters
        lowcut=30 #43
        highcut=300 #260
        order = 4

        # bandpass filter coefficients
        # do bandpass as last filter. #HELP! 
        #nyq = 0.5*fs
        #low = lowcut / nyq
        #high = highcut / nyq
        #bb, ab = butter(order, [low, high], btype='band')
        #strain_bp_1 = filtfilt(bb, ab, strain_bp)

        # number of sample for the fast fourier transform:
        NFFT = 1*fs
        Pxx, freqs = mlab.psd(strain_fil, Fs = fs, NFFT = NFFT)
        
        # We will use interpolations of the ASDs computed above for whitening:
        psd = interp1d(freqs, Pxx)      #substitute with interpolating function (np.fit, curve_fit a/f**i+c+bf**j )
        
        
        ###### interpolating function (not working as data isn't properly notched imo)
        def f_fit(x,a,b,c,i,j):
            return a/(x**i)+c+b*x**j
        
        psd_params, psd_cov = curve_fit(f_fit,freqs[50:300], Pxx[50:300],bounds=([-1.,-1.,-1.E-25,0.,0.],[1.,1.,1.E-20,1.5,1.5 ]))
        
        print psd_params
        #########

        #Should really use analytic fit to PSD now that it is notched

        # now whiten the data 
        #strain_whiten = whiten(strain_bp,psd,dt)
        
        white_hf = self.whiten(strain_fil,psd,dt)
        white_hf_bp = self.buttering(white_hf)*white_hf
        #print white_hf
        
        # bandpass filter parameters
        #lowcut=20 #43
        #highcut=300 #260
        #order = 4

        # bandpass filter coefficients
        # do bandpass as last filter
        #nyq = 0.5*fs
        #low = lowcut / nyq
        #high = highcut / nyq
        #bb, ab = butter(order, [low, high], btype='band')
        #strain_bp = filtfilt(bb, ab, strain_whiten)
    
        return white_hf_bp, white_hf
    

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

        # TODO: may want to do this in frequency space
        #strain_x = strain_H1*strain_L1
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
            print 'split ', n_split, len(ctime)/fs
            ctime = np.array_split(ctime, n_split)
            strain_H1 = np.array_split(strain_H1, n_split)
            strain_L1 = np.array_split(strain_L1, n_split)
            return ctime, strain_H1, strain_L1 #strain_x
            
        else:
            # add dummy dimension
            n_split = 1
            ctime = ctime[None,...]
            strain_H1 = strain_H1[None,...]
            strain_L1 = strain_L1[None,...]
            return ctime, strain_H1, strain_L1 #strain_x
    
    
    #return ctime, strain_H1, strain_L1 #strain_x


    # ********* Projector *********
    # returns p = {lm} map of inverse-noise-filtered time-stream
       
    def summer(self, ct_split, s, freq):        
        nside=self._nside
        lmax=self._lmax
        
        sum_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
        hit_lm = np.zeros_like(sum_lm)
        
        #if you input s_split = 1, (maybe) it returns simply the projection operator 
        
        npix = self.npix
        
        mid_idx = int(len(ct_split)/2)
    
        # get quaternions for H1->L1 baseline (use as boresight)
        q_b = self.Q.azel2bore(np.degrees(self.az_b), np.degrees(self.el_b), None, None, np.degrees(self.H1_lon), np.degrees(self.H1_lat), ct_split[mid_idx])
        # get quaternions for bisector pointing (use as boresight)
        q_n = self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid), np.degrees(self.latMid), ct_split[mid_idx])[0]

        pix_b, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True) #spin-2
        
        p = pix_b          
        quat = q_n
        #s = np.average(s_filt)
    
        # polar angles of baseline vector
        theta_b, phi_b = hp.pix2ang(nside,p)
        
        #print theta_b, phi_b
        
        # rotate gammas
        # TODO: will need to oversample here
        # i.e. use nside > nside final map
        # TODO: pol gammas
        rot_m_array = self.rotation_pix(np.arange(npix), quat) #rotating around the bisector of the gc 
        gammaI_rot = self.gammaI[rot_m_array]

        # Expand rotated gammas into lm
        glm = hp.map2alm(gammaI_rot, lmax, pol=False)              
        
        # sum over lp, mp
        for l in range(lmax+1):
            for m in range(l+1):
                idx_lm = hp.Alm.getidx(lmax,l,m)
                for idx_f, f in enumerate(freq):    #hits = 0 here maybe..?
                #print idx_f
                    hits = 0
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
                                                    self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp]*s[idx_f]) 
                                                    
                                hit_lm[idx_lm] += ((-1)**mpp*(0+1.j)**lpp*self.dfreq_factor(f,lpp)
                                *sph_harm(mpp, lpp, theta_b, phi_b)*
                                                    glm[hp.Alm.getidx(lmax,lp,mp)]*np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*
                                                    self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp])
                                hits+=1

        norm = np.abs(np.sum(np.abs(hit_lm))) #maybe?
        print norm
        print np.abs(hit_lm/norm), np.sum(np.abs(hit_lm/norm))
        exit()
        #print np.divide(sum_lm,hit_lm)
        return sum_lm/norm #np.divide(sum_lm,hit_lm) #

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
        
    def projector(self,ctime, strain_H1, strain_H2, freq_x_coar, cf=10000.):
        print 'proj run'    
        nside=self._nside
        lmax=self._lmax
            
        npix = self.npix
        data_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
        hits = 0.
        for idx_t, (ct_split, s_1, s_2, freq) in enumerate(zip(ctime, strain_H1, strain_H2, freq_x_coar)): 
            
            s = s_1*np.conj(s_2)/(60.*4096./cf) ## div by number of new samples in a 1minute segment
            
            data_lm += self.summer(ct_split, s, freq)
            
            hits+=1.
        
        return data_lm/hits

    # ********* Scanner *********
    #to use scanner, re-check l, m ranges and other things. otherwise use scanner_1    
    def scanner(self,ct_split, p_split_1_t, p_split_2_t,freq,cf=10000,data_lm = 1.): #scanner(ctime_array, idx_t, p_split_1[idx_t], p_split_2, freq_coar_array)
        nside=self._nside
        lmax=self._lmax 
            
        npix = self.npix
        
        scanned_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
        hit_lm = np.zeros(len(scanned_lm))
        hits = 0.

        mid_idx = int(len(ct_split)/2)
        
        q_b = self.Q.azel2bore(np.degrees(self.az_b), np.degrees(self.el_b), None, None, np.degrees(self.H1_lon), np.degrees(self.H1_lat), ct_split[mid_idx])
        q_n = self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid), np.degrees(self.latMid), ct_split[mid_idx])[0]
        
        p, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True) 
        
        quat = q_n
        
        theta_b, phi_b = hp.pix2ang(nside,p)
        
        rot_m_array = self.rotation_pix(np.arange(npix), quat) 
        gammaI_rot = self.gammaI[rot_m_array]

        glm = hp.map2alm(gammaI_rot, lmax, pol=False)
        
        weight = np.divide(1.,p_split_1_t*p_split_2_t)/(60.*4096./cf)  #np.ones_like(p_split_1_t) #
        
        for idx_f, f in enumerate(freq):
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
                                scanned_lm[idx_lm] += ((-1)**mpp*(0+1.j)**lpp*sph_harm(mpp, lpp, theta_b, phi_b)*self.dfreq_factor(f,lpp)
                                *glm[hp.Alm.getidx(lmax,lp,mp)]*np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp])*data_lm*weight[idx_f]
                                hit_lm[idx_lm] += 1.
                                hits += 1.
        return np.divide(scanned_lm,hit_lm) #hp.alm2map(scanned_lm/hits,nside,lmax=lmax)


    def scanner_1(self,ctime, idx_t, p_split_1_t, p_split_2,freq,data_lm = 1.): #scanner(ctime_array, idx_t, p_split_1[idx_t], p_split_2, freq_coar_array)
    
        nside=self._nside
        lmax=self._lmax 
            
        npix = self.npix
        
        scanned_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
        hit_lm = np.zeros(len(scanned_lm))
        hits = 0.
        
        #        for idx_t, ct_split in enumerate(ctime):
        for idx_t_p, ct_split_p in enumerate(ctime):
            
            print idx_t_p
            
            mid_idx = int(len(ct_split_p)/2)
            
            q_b = self.Q.azel2bore(np.degrees(self.az_b), np.degrees(self.el_b), None, None, np.degrees(self.H1_lon), np.degrees(self.H1_lat), ct_split_p[mid_idx])
            q_n = self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid), np.degrees(self.latMid), ct_split_p[mid_idx])[0]
            
            p, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True) 
            
            quat = q_n
            
            theta_b, phi_b = hp.pix2ang(nside,p)
            
            rot_m_array = self.rotation_pix(np.arange(npix), quat) 
            gammaI_rot = self.gammaI[rot_m_array]

            glm = hp.map2alm(gammaI_rot, lmax, pol=False)
            
            weight = np.divide(1.,p_split_1_t*p_split_2[idx_t_p])
            
            for idx_f, f in enumerate(freq[idx_t_p]):
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
                                    scanned_lm[idx_lm] += ((-1)**mpp*(0+1.j)**lpp*sph_harm(mpp, lpp, theta_b, phi_b)*self.dfreq_factor(f,lpp)
                                    *glm[hp.Alm.getidx(lmax,lp,mp)]*np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp])*data_lm*weight[idx_f]
                                    hit_lm[idx_lm] += 1.
                                    hits += 1.
        return scanned_lm/hits #hp.alm2map(scanned_lm/hits,nside,lmax=lmax)


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
    
    # ********* Projector Matrix *********

'''
    Nt = len(strain)
    Nt = lf.bestFFTlength(Nt)
    plt.figure()
    #plt.axis([1200.,60000., -1E-14, 1E-14])
    plt.plot(np.abs(np.fft.rfft(strain_fil[:Nt])/Nt),'r',label='fft')
    pylab.ylim([0.,1E-21])
    plt.savefig('fft.png')
    plt.figure()
    plt.plot(np.sqrt(Pxx),'g',label='Pxx')
    pylab.ylim([0.,1E-21])
    plt.legend()
    plt.savefig('Pxx.png')
'''        
    # ********* Scanner Matrix *********
    
#    def A_tp(self, ct_split):
        