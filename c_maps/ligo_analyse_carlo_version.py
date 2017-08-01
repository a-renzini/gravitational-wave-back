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
import qpoint as qp
import healpy as hp
from camb.bispectrum import threej
import quat_rotation as qr

# from Arianna
import OverlapFunctsSrc as ofs
#from response_function import rotation_pix

# LIGO-specific readligo.py 
import readligo as rl
import ligo_filter as lf
from gwpy.time import tconvert

def rotation_pix(Q,m_array,n): #rotates string of pixels m around QUATERNION n
    nside = hp.npix2nside(len(m_array))
    dec_quatmap,ra_quatmap = hp.pix2ang(nside,m_array) #
    quatmap = Q.radecpa2quat(np.rad2deg(ra_quatmap), np.rad2deg(dec_quatmap-np.pi*0.5), 0.*np.ones_like(ra_quatmap)) #but maybe orientation here is actually the orientation of detector a, b? in which case, one could input it as a variable!
    quatmap_rotated = np.ones_like(quatmap)
    i = 0
    while i < len(m_array): #used to be lenmap
        quatmap_rotated[i] = qr.quat_mult(n,quatmap[i])
        i+=1
    quatmap_rot_pix = Q.quat2pix(quatmap_rotated,nside)[0] #rotated pixel list (pols are in [1])
    return quatmap_rot_pix

def dfreq_factor(f,ell,b,alpha=3.,H0=68.0,f0=100.):
    # f : frequency (Hz)
    # ell : multipole
    # alpha: spectral index
    # b: baseline length (m)
    # f0: pivot frequency (Hz)
    # H0: Hubble rate today (km/s/Mpc

    km_mpc = 3.086e+19 # km/Mpc conversion
    c = 3.e8 # speed of light 
    #fac = 8.*np.pi**3/3./H0**2 * km_mpc**2 * f**3*(f/f0)**(alpha-3.) * spherical_jn(ell, 2.*np.pi*f*b/c)
    fac = (f/f0)**(alpha-3.) * spherical_jn(ell, 2.*np.pi*f*b/c)#*f**3
    # add band pass and notches here
 
    return fac

def freq_factor(ell,b,fmin,fmax,alpha=3.,H0=68.0,f0=100.):
    return integrate.quad(dfreq_factor,fmin,fmax,args=(ell,b))[0]
    
def vec2azel(v1,v2):
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

def midpoint(lat1,lon1,lat2,lon2):
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


# sampling rate:
fs = 4096
ligo_data_dir = '/Users/pai/Data'
filelist = rl.FileList(directory=ligo_data_dir)

# Configuration: radians and metres, Earth-centered frame
H1_lon = -2.08405676917
H1_lat = 0.81079526383
H1_elev = 142.554
H1_vec = np.array([-2.16141492636e+06, -3.83469517889e+06, 4.60035022664e+06]) 

L1_lon = -1.58430937078
L1_lat = 0.53342313506
L1_elev = -6.574
L1_vec = np.array([-7.42760447238e+04, -5.49628371971e+06, 3.22425701744e+06])

V1_lon = 0.18333805213
V1_lat = 0.76151183984
V1_elev = 51.884
V1_vec = np.array([4.54637409900e+06, 8.42989697626e+05, 4.37857696241e+06])

# work out viewing angle of baseline H1->L1
az_b, el_b, baseline_length = vec2azel(H1_vec,L1_vec)
# position of mid point and angle of great circle connecting to observatories
latMid,lonMid, azMid = midpoint(H1_lat,H1_lon,L1_lat,L1_lon)

nside = 32
lmax = 16
npix = hp.nside2npix(nside)
data_map = np.zeros(npix)
hits_map = np.zeros_like(data_map)

# cache 3j symbols
threej_0 = np.zeros((2*lmax,2*lmax,2*lmax))
threej_m = np.zeros((2*lmax,2*lmax,2*lmax,2*lmax,2*lmax))
for l in range(lmax):
    for m in range(l):
        for lp in range(lmax):
            lmin0 = np.abs(l - lp)
            lmax0 = l + lp
            threej_0[lmin0:lmax0+1,l,lp] = threej(l, lp, 0, 0)
            for mp in range(lp):
                # remaining m index
                mpp = m+mp
                lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                lmax_m = l + lp
                threej_m[lmin_m:lmax_m+1,l,lp,m,mp] = threej(l, lp, m, mp)

# prepare lookup of frequency factor
fl = [freq_factor(l,baseline_length,0.01,300.) for l in range(lmax*4)]
 
# calculate overlap functions
# TODO: integrate this with general detector table
theta, phi = hp.pix2ang(nside,np.arange(npix)) 
gammaI = ofs.gammaIHL(theta,phi)
gammaQ = ofs.gammaQHL(theta,phi)
gammaU = ofs.gammaUHL(theta,phi)
gammaV = ofs.gammaVHL(theta,phi)
m_array = np.arange(npix)

# These will accumulate from timestream 
data_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
out_lm = np.zeros_like(data_lm)
hit_lm = np.zeros(len(data_lm))

# create qpoint object
Q = qp.QPoint(accuracy='low', fast_math=True, mean_aber=True)#, num_threads=1)

# define start and stop time to search
# in GPS seconds
start = 931035615 #931079472
stop  = 971622015 #931086336 #
#start = 931079472
#stop  = 931086336
#start = 931200000
#stop = 931300000

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

hits = 0.
# Now loop over all segments found
for sdx, (begin, end) in enumerate(zip(segs_begin,segs_end)):    

    # load data
    strain_H1, meta_H1, dq_H1 = rl.getstrain(begin, end, 'H1', filelist=filelist)
    strain_L1, meta_L1, dq_L1 = rl.getstrain(begin, end, 'L1', filelist=filelist)

    # TODO: may want to do this in frequency space?
    strain_x = strain_H1*strain_L1
    
    # Figure out unix time for this segment
    # This is the ctime for qpoint
    utc_begin = tconvert(meta_H1['start']).replace(tzinfo=pytz.utc)
    utc_end = tconvert(meta_H1['stop']).replace(tzinfo=pytz.utc)
    ctime = np.arange((utc_begin - epoch).total_seconds(),(utc_end - epoch).total_seconds(),meta_H1['dt'])

    # discard very short segments
    if len(ctime)/fs < 16: continue
    print utc_begin, utc_end

    # if long then split into sub segments
    if len(ctime)/fs > 120:
        # split segment into sub parts
        # not interested in freq < 40Hz
        n_split = np.int(len(ctime)/(60*fs))
        print 'split ', n_split, len(ctime)/fs
        ctime = np.array_split(ctime, n_split)
        strain_x = np.array_split(strain_x, n_split)
    else:
        # add dummy dimension
        n_split = 1
        ctime = ctime[None,...]
        strain_x = strain_x[None,...]
        
    for idx, (ct_split, s_split) in enumerate(zip(ctime, strain_x)):
        
        # get quaternions for H1->L1 baseline (use as boresight)
        q_b = Q.azel2bore(np.degrees(az_b), np.degrees(el_b), None, None, np.degrees(H1_lon), np.degrees(H1_lat), ct_split)
        # get quaternions for bisector pointing (use as boresight)
        #q_n = Q.azel2bore(azMid, 90.0, None, None, np.degrees(lonMid), np.degrees(latMid), ct_split)
        q_n = Q.azel2bore(0., 90.0, None, None, np.degrees(lonMid), np.degrees(latMid), ct_split)
        
        pix_b, s2p, c2p = Q.quat2pix(q_b, nside=nside, pol=True) #spin-2

        print '{}/{} {} seconds'.format(idx,n_split,len(ct_split)/fs)
        
        # Filter the data
        #strain_H1 = lf.whitenbp_notch(strain_H1)
        #strain_L1= lf.whitenbp_notch(strain_L1)
        s_filt = lf.whitenbp_notch(s_split)

        # This is the 'projection' side of mapping equation
        # z_p = (A_tp)^T N_tt'^-1 d_t'= A_pt  N_tt'^-1 d_t'
        # It takes a timestream, inverse noise filters and projects onto
        # pixels p (here p are actually lm)
        # this is the 'dirty map'

        # The sky map is obtained by
        # s_p = (A_pt N_tt'^-1 A_t'p')^-1  z_p'
        
        #sum over time
        #for tidx, (p, s, quat) in enumerate(zip(pix_b,s_filt,q_n)):

        # average over sub segment and use
        # middle of segment for pointing etc.
        mid_idx = len(pix_b)/2
        p = pix_b[mid_idx]
        quat = q_n[mid_idx]
        s = np.average(s_filt)
 
        # polar angles of baseline vector
        theta_b, phi_b = hp.pix2ang(nside,p)

        # rotate gammas
        # TODO: will need to oversample here
        # i.e. use nside > nside final map
        # TODO: pol gammas
        rot_m_array = rotation_pix(Q, np.arange(npix), quat)
        gammaI_rot = gammaI[rot_m_array]
        
        # Expand rotated gammas into lm
        glm = hp.map2alm(gammaI_rot, lmax, pol=False)

        # sum over lp, mp
        for l in range(lmax):
            for m in range(l):
                #print l, m
                idx_lm = hp.Alm.getidx(lmax,l,m)
                for lp in range(lmax):
                    for mp in range(lp):
                        # remaining m index
                        mpp = m+mp
                        lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                        lmax_m = l + lp
                        for idxl, lpp in enumerate(range(lmin_m,lmax_m+1)):
                            data_lm[idx_lm] += ((-1)**mpp*(0+1.j)**lpp*fl[lpp]*sph_harm(mpp, lpp, theta_b, phi_b)*
                                                glm[hp.Alm.getidx(lmax,lp,mp)]*np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*
                                                threej_0[lpp,l,lp]*threej_m[lpp,l,lp,m,mp]*s)
                            hit_lm[idx_lm] += 1.
                            hits += 1.

        #print data_lm

    out = hp.alm2map(data_lm/hits,nside,lmax=lmax)
    #out = np.copy(data_map)
    #out[hits_map > 0] /= hits_map[hits_map > 0]
    #out[hits_map==0.] = hp.UNSEEN
    hp.mollview(out)
    plt.savefig('map%s.pdf' %sdx)
    hp.write_map("map.fits",out)
     
    
#data_map[hits_map > 0] /= hits_map[hits_map > 0]
#data_map[hits_map == 0.] = hp.UNSEEN
hp.mollview(out)
plt.savefig('map%s.pdf' %sdx)
hp.write_map("map.fits",data_map)

exit()