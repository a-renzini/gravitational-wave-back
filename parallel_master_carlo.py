import numpy as np
import qpoint as qp
import healpy as hp
import pylab
#import ligo_analyse_class as lac
from ligo_analyse_class import Ligo_Analyse
import readligo as rl
import ligo_filter as lf
import matplotlib as mlb
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
import MapBack as mb  #################
from mpi4py import MPI
ISMPI = True
#if mpi4py not present: ISMPI = False

import os
import sys

data_path = sys.argv[1]
out_path =  sys.argv[2]

try:
    sys.argv[3]
except (NameError, IndexError):
    checkpoint = None
else:
    checkpoint = True
    checkfile_path = sys.argv[3]

    
if os.path.exists(data_path):
    print 'the data its in the ' , data_path
    # file exists                                                                                                             
if os.path.exists(out_path):
    print 'output goes to ' , out_path
    
###############                                                                                                               

def split(container, count):
    """                                                                                                                       
    Simple function splitting a container into equal length chunks.                                                           
    Order is not preserved but this is potentially an advantage depending on                                                  
    the use case.                                                                                                             
    """
    return [container[_i::count] for _i in range(count)]

###############                                                                                                               

EPSILON = 1E-24

if ISMPI:

    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    myid = comm.Get_rank()


else:
    comm = None
    nproc = 1
    myid = 0

print 'myid: {} of {}'.format(myid,nproc)

####################################################################                                                          
print '++++++++++++++++++++++++++'
print '=========================='
print '++++++++++++++++++++++++++'
print '=========================='
print (time.strftime("%H:%M:%S")), (time.strftime("%d/%m/%Y"))
print '++++++++++++++++++++++++++'
print '=========================='
print '++++++++++++++++++++++++++'
print '=========================='

# sampling rate:                                                                                                              
fs = 4096

ligo_data_dir = data_path  #can be defined in the repo                                                                        
filelist = rl.FileList(directory=ligo_data_dir)

nside_in = 32
nside_out = 8
#nside_in = nside_out
npix_in = hp.nside2npix(nside_in)
npix_out = hp.nside2npix(nside_out)
lmax = 4
sim = False

#map in
alm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=np.complex)
idx = hp.Alm.getidx(lmax,4,2)
alm[idx] = 1.+ 0.j
map_in = hp.alm2map(alm,nside=nside_in)

#vectors to pixels in/out resolution
vec_p_in = hp.pix2vec(nside_in,np.arange(npix_in))
vec_p = hp.pix2vec(nside_out,np.arange(npix_out))
hp.mollview(map_in)
plt.savefig('map_in.pdf' )


#INTEGRATING FREQS:                                                                                                           
low_f = 80.
high_f = 300.
low_cut = 60.
high_cut = 200.

    
#DETECTORS
dects = ['H1','L1','V1']
ndet = len(dects)
nbase = int(ndet*(ndet-1)/2)
 
#create object of class:
run = mb.Telescope(nside_in,nside_out,lmax, fs, low_f, high_f, dects)

# Store first baseline gamma
gammaI = run.gammaI[0]

# define start and stop time to search
# in GPS seconds
start = 931035615 #S6 start GPS
stop  = 931122015
#stop  = 971622015  #S6 end GPS
start = 1126051217 #O1 start
stop  = 1198101517 

###########################UNCOMMENT ME#########################################

print 'flagging the good data...'

segs_begin, segs_end = run.flagger(start,stop,filelist)

ctime_nproc = []
strain1_nproc = []
strain2_nproc = []
b_pixes = []

if myid == 0:
    z_p_glob  = np.zeros(npix_out)
    M_pp_glob = np.zeros((npix_out,npix_out))

    counter = 0
    conds = []
    if checkpoint  == True:
        checkdata = np.load(checkfile_path)
        Z_lm += checkdata['Z_lm']
        M_lm_lpmp += checkdata['M_lm_lpmp']
        S_lm = None
        counter = checkdata['counter']
        conds = checkdata['conds']

else:
    z_p_glob = None
    M_pp_glob = None
    counter = 0

print 'segmenting the data...'
#create empty lm objects:
z_p = np.zeros(npix_out)
M_pp = np.zeros((npix_out,npix_out))

for sdx, (begin, end) in enumerate(zip(segs_begin,segs_end)):

    print '{} of {}'.format(sdx,len(segs_begin))
    n=sdx+1

    ctime, strain_H1, strain_L1 = run.segmenter(begin,end,filelist)

    if len(ctime) < 2 : continue
    
    idx_block = 0

    while idx_block < len(ctime):
        ctime_nproc.append(ctime[idx_block])
        strain1_nproc.append(strain_H1[idx_block])
        strain2_nproc.append(strain_L1[idx_block])
        
        if len(ctime_nproc) == nproc:  

            idx_list = np.arange(nproc)

            if myid == 0:
                my_idx = np.split(idx_list, nproc)  #probably redundant .. could just say my_ctime = ctime[myid]

            else:
                my_idx = None


            if ISMPI:
                my_idx = comm.scatter(my_idx)
                my_ctime = ctime_nproc[my_idx[0]]
                my_h1 = strain1_nproc[my_idx[0]]
                my_l1 = strain2_nproc[my_idx[0]]


            #print 'filtering, ffting & saving the strains...'

            Nt = len(my_h1)
            df_h1 = np.fft.rfft(my_h1,norm='ortho')
            df_l1 = np.fft.rfft(my_l1,norm='ortho')

            pf_h1 = df_h1*np.conj(df_h1)
            pf_l1 = df_l1*np.conj(df_l1)
            pf = np.sqrt(pf_h1*pf_l1)
            
            freqs = np.fft.rfftfreq(Nt, 1./fs)

            df = df_h1*np.conj(df_l1)
 
            #print df
            #print pf

            delta_freq = 1.*fs/len(freqs)
            window = np.ones_like(freqs)
            mask = (freqs < high_cut) & (freqs > low_cut)
            
            #print window[(freqs < high_cut) & (freqs > low_cut)]
            E_freq = np.ones_like(freqs)
            
            # get gamma rotated to this time segment
            pix_bs, q_ns, pix_ns = run.geometry(my_ctime)

            # Rotate gamma 
            #gammaI = run.gammaI[0]
            rot_m_array = run.rotation_pix(np.arange(npix_in), q_ns[0])
            gammaI_rot = run.gammaI[0][rot_m_array]
            # Degrade to out resolution
            gammaI_rot_ud = hp.ud_grade(gammaI_rot,nside_out = nside_out)           
            
            vec_b = hp.pix2vec(nside_out,pix_bs[0])
            bdotp = 2.*np.pi*np.dot(vec_b,vec_p)*run.R_earth/3.e8

            #for simulated frequency data
            bdotp_in = 2.*np.pi*np.dot(vec_b,vec_p_in)*run.R_earth/3.e8
            #pf = np.ones_like(pf)
            #df *= 0.

            # Mask to required frequency range only
            window = window[mask]
            E_freq = E_freq[mask]
            freqs = freqs[mask]
            df = df[mask]
            pf = pf[mask]
            
            print 'Scanning sky...'
            for idx_f in range(len(freqs)):
                df[idx_f] = 4.*np.pi/npix_in * delta_freq*np.sum(window[idx_f] * E_freq[idx_f] * gammaI_rot[:] * map_in[:]\
                                                                 *(np.cos(bdotp_in[:]*freqs[idx_f]) + np.sin(bdotp_in[:]*freqs[idx_f])*1.j)) 

            print 'Projecting data...'
            for ip in range(npix_out):
                z_p[ip] += 8.*np.pi/npix_out * delta_freq*np.sum(window[:] * E_freq[:]/ pf[:]**2 * gammaI_rot_ud[ip]\
                                                                *(np.cos(bdotp[ip]*freqs[:])*np.real(df[:]) - np.sin(bdotp[ip]*freqs[:])*np.imag(df[:]))) 
                for jp in range(ip,npix_out):
                    val = 2.*(4.*np.pi)**2/npix_out**2 * delta_freq**2 * np.sum(window[:]**2 * E_freq[:]**2/ pf[:]**2 \
                                                                                                     * gammaI_rot_ud[ip] * gammaI_rot_ud[jp]\
                                                                                                     *(np.cos((bdotp[ip]-bdotp[jp])*freqs[:]) )) 
                    M_pp[ip,jp] += val
                    if ip!= jp : M_pp[jp,ip] += val


            print 'Solving...'
            #lam, R = np.linalg.eigh(M_pp)

            #mask = lam < 1.e-5*lam[0]
            
            #I = np.eye(len(lam))
            #inv_lam = 1/lam[..., None] * I[None, ...]
            #Rt = R.swapaxes(-1, -2)

            # optimized matrix multiplication
            #Rinv = np.einsum('...ij,...jk->...ik', R, inv_lam)
            #M_pp_inv = np.einsum('...ik,...kl->...il', Rinv, Rt)#.swapaxes(0, -1)
            #print lam
            #m_p = R[-1]
            #print 'Condition nuber: {}'.format(np.amax(np.abs(lam))/np.amin(np.abs(lam)))
            M_pp_inv = np.linalg.pinv(M_pp,rcond=1.e-5)
            m_p = np.einsum('...ik,...k->...i', M_pp_inv, z_p)
            #m_p = np.linalg.solve(M_pp,z_p,rcond=1.e-8)                

            hp.mollview(z_p)
            plt.savefig('z_p.pdf' )
            hp.mollview(m_p)
            plt.savefig('m_p.pdf' )
            #hp.mollview(gammaI)
            #plt.savefig('gamma.pdf' )
            hp.mollview(gammaI_rot)
            plt.savefig('gamma_rot.pdf' )
            
            """
            
            Nt = len(my_h1)
            Nt = lf.bestFFTlength(Nt)
            
            freqs = np.fft.rfftfreq(2*Nt, 1./fs)
            freqs = freqs[:Nt/2+1]
            
            mask = (freqs>low_f) & (freqs < high_f)
            
            strains = (my_h1,my_l1)
            strains_copy = (my_h1.copy(),my_l1.copy()) #calcualte psds from these

                    ###########################


            if sim == True:
                print 'generating...'
                h1_in = my_h1.copy()
                l1_in = my_l1.copy()
                strains_in = (h1_in,l1_in)
                #print strains_in
                strains = run.injector(strains_in,my_ctime,low_cut,high_cut, sim)[0]
                #print len(strains)
                # plt.figure()
                # plt.plot((strains[0]))
                # plt.savefig('fakestreamsinv.pdf')
                # f = open('fakestreamsinv.txt', 'w')
                # for (i,x) in enumerate(strains[0]):
                #     print >>f, i, '     ', x
                # f.close()
                #pass the noisy strains to injector got the psds
            psds = run.injector(strains_copy,my_ctime,low_cut,high_cut)[1]
            #strains are the new generated strains
            
            #print 'std of corr. t_stream: ', np.std(strains[0]*strains[1])
            
            
            strains_f = []
            psds_f = []
            strains_w = []

            for i in range(ndet):
                strains_f.append(run.filter(strains[i], low_cut,high_cut,psds[i]))
                psds_f.append(psds[i](freqs)*fs**2) 
                #psds_f[i] = np.ones_like(psds_f[i])
                strains_w.append(strains_f[i]/(psds_f[i]))
                    
            
            #print strains_f[0][mask]*np.conj(strains_f[1])[mask]
            #print np.average(strains_f[0][mask]*np.conj(strains_f[1])[mask])
            

            '''
            now strains_w, etc are pairs of 60s segments of signal, in frequency space.
            '''


                #print '+++'
                #print run.sim_tstream(ctime[0],1.,1.,freqs)
            print 'filtering done'
                #Integrate over frequency in the projector

            ####################################################################

            #proj_lm = np.array([np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)]*len(ctime)) #why *len_ctime?

            print 'running the projector, obtaining a dirty map'

            pix_bs = run.geometry(my_ctime)[0]
            q_ns = run.geometry(my_ctime)[1]
            pix_ns = run.geometry(my_ctime)[2]
            
            print pix_bs
                        
            for i in range(len(pix_bs)):
                b_pixes.append(pix_bs[i])
            
            
            z_lm = run.projector(my_ctime,strains_w,freqs,pix_bs, q_ns)
            
            
            if myid == 0:
                z_buffer = np.zeros_like(z_lm)
            else:
                z_buffer = None

            if ISMPI:
                comm.barrier()
                comm.Reduce(z_lm, z_buffer, root = 0, op = MPI.SUM)
                if myid ==0: counter += nproc

            else:
                z_buffer += z_lm
                counter += 1
            #print '----'
            #print 'z_lm', z_lm
            #print 'buffer', z_buffer
            #print 'counter',counter
            #print '----'


            if myid == 0:

                print 'this is id 0'
                Z_lm += z_buffer
                
                print '+++'
                print counter, 'mins analysed.'
                print '+++'

                # for idx_t, ct_split in enumerate(ctime):
                #     ones = [1.]*len(freqs)
                #     proj_lm[idx_t] = run.summer(ctime[idx_t],ones,freqs)

                #dirty_map_lm = hp.alm2map(np.sum(dt_lm,axis = 0),nside,lmax=lmax)
                                    
            
                print 'building M^-1:'

            my_M_lm_lpmp += run.M_lm_lpmp_t(my_ctime, psds_f,freqs,pix_bs,q_ns)

            cond = np.linalg.cond(my_M_lm_lpmp)

            if myid == 0:
                M_lm_lpmp_buffer = np.zeros_like(my_M_lm_lpmp)
                conds_array = np.zeros(nproc)
            else:
                M_lm_lpmp_buffer = None
                conds_array = None

            if ISMPI:
                comm.barrier()
                comm.Reduce(my_M_lm_lpmp, M_lm_lpmp_buffer, root = 0, op = MPI.SUM)
                comm.Gather(cond, conds_array, root = 0)

            else:
                M_lm_lpmp_buffer += my_M_lm_lpmp
                conds.append(cond)

            if myid == 0:
                M_lm_lpmp += np.real(M_lm_lpmp_buffer)
                np.append(conds,conds_array)

                #print 'M is ', len(M_lm_lpmp), ' by ', len(M_lm_lpmp[0])

                print 'Inverting M...'

                #### SVD

                M_lm_lpmp = np.real(M_lm_lpmp)
                M_inv = np.linalg.pinv(M_lm_lpmp)   #default:  for cond < 1E15

                print 'the matrix has been inverted!'


                ################################################################

                S_lm = np.array(np.dot(M_inv,Z_lm)) #fully accumulated maps!
                #S_lm+= s_lm

                #print len(s_lm)
                #print s_lm

                #dt_tot = np.sum(dt_lm,axis = 0)
                #print 'dt total:' , len(dt_tot.real)
                #print dt_tot
                
                if counter % (nproc*20) == 0:    ## *10000
                    
                    f = open('%s/M%s.txt' % (out_path,counter), 'w')
                    print >>f, 'sim = ', sim
                    print >>f, M_lm_lpmp
                    print >>f, '===='
                    print >>f, M_inv
                    print >>f, '===='                    
                    print >>f, np.linalg.eigh(M_lm_lpmp)
                    print >>f, '===='
                    print >>f, cond
                    print >>f, '===='
                    print >>f, np.dot(M_lm_lpmp,M_inv),np.identity(len(M_lm_lpmp))
                    f.close()
                    
                    
                    dirty_map = hp.alm2map(Z_lm,nside_out,lmax=lmax)
                    S_p = hp.alm2map(S_lm,nside_out,lmax=lmax)
                    
                    
                    fig = plt.figure()
                    hp.mollview(dirty_map)
                    plt.savefig('%s/dirty_map%s.pdf' % (out_path, counter))

                    fig = plt.figure()
                    hp.mollview(S_p)
                    plt.savefig('%s/S_p%s.pdf' % (out_path,counter))
                    
                    
                    np.savez('%s/checkfile%s.npz' % (out_path,counter), Z_lm=Z_lm, M_lm_lpmp=M_lm_lpmp, counter = counter, conds = conds )
                    
                    print 'saved dirty_map, clean_map and checkfile @ min', counter
                    
                    falm = open('%s/alms%s.txt' % (out_path,counter), 'w')
                    print >> falm, S_lm  
                    for l in range(lmax+1):
                        idxl0 =  hp.Alm.getidx(lmax,l,0)
                         
                        almbit = 0.
                        for m in range(l+1):
                            idxlm =  hp.Alm.getidx(lmax,l,m)
                            almbit +=(2*S_lm[idxlm])*np.conj(S_lm[idxlm])/(2*l+1)
                        
                        print >> falm, almbit - S_lm[idxl0]*np.conj(S_lm[idxl0])/(2*l+1)
                        print >> falm, np.average(S_p)
                    print >> falm, 'end.'
                    falm.close()
                    
                    fig = plt.figure()
                    hp.mollview(np.zeros_like(dirty_map))
                    hp.visufunc.projscatter(hp.pix2ang(nside_out,b_pixes))
                    plt.savefig('%s/b_pixs%s.pdf' % (out_path,counter))
                    
                    #exit()
                    
                    #if counter == 40:  exit()

                    """
                #################################################    
                #################################################    
                #################################################    
            ctime_nproc = []
            strain1_nproc = []
            strain2_nproc = []
        
        #print idx_block    
        idx_block += 1
        #if idx_block == 1400: print S/N

"""
if myid == 0:

    hp.mollview(hp.alm2map(Z_lm,nside_out,lmax=lmax))
    plt.savefig('%sZ_p%s.pdf' % (out_path,counter))

    hp.mollview(hp.alm2map(S_lm,nside_out,lmax=lmax))
    plt.savefig('%sS_p%s.pdf' % (out_path,counter))
    

    
    
    ############# using the decorrelator instead: #########
    #
    #
    # BIG_M = run.decorrelator(ctime, freqs_x_coar, p_split_1, p_split_2)
    # print 'BIG M is ', len(BIG_M), ' by ', len(BIG_M[0])
    # BIG_M_inv = np.linalg.inv(BIG_M)
    #
    # s_lm = np.array(np.dot(BIG_M_inv,z_lm))
    # s_p = hp.alm2map(s_lm,nside,lmax=lmax)
    #
    # hp.mollview(s_p)
    # plt.savefig('s_BIG_M_p.pdf')
    
    
    #plt.figure()
    #plt.axis([0,10000, 0.001, 10.]) 
    #plt.loglog()
    #plt.plot(fitted,label = 'fitted psd')
    #pylab.xlim([50.,500.])
    #pylab.ylim([0.,1E-43])
    #plt.xlabel('')
    #plt.ylabel('')
    #plt.legend()
    #plt.savefig('datapow.png')


''' 
print 'building M^-1:'

M_lm_lpmp =[]
    
print '1. scanning...'

scan_lm = []


for idx_t in range(len(ctime)):
        scan_lm.append(run.scanner(ctime[idx_t], p_split_1[idx_t],p_split_2[idx_t],freqs))

print '2. projecting...'

for idx_lm in range(hp.Alm.getidx(lmax,lmax,lmax)+1):
    
    M_lpmp = np.zeros(len(proj_lm[0]),dtype=complex)
    #print idx_lm
    if proj_lm[0][idx_lm]>EPSILON or proj_lm[0][idx_lm]<-EPSILON:
        for idx_t in range(len(ctime)):
            scan = scan_lm[idx_t]
            proj = proj_lm[idx_t][idx_lm]
            M_lpmp += np.conj(proj)*scan
    M_lm_lpmp.append(M_lpmp)

print 'M is ', len(M_lm_lpmp), ' by ', len(M_lm_lpmp[0])

print M_lm_lpmp

print '3. inverting...'

M_inv = np.linalg.inv(M_lm_lpmp)

print 'the matrix has been inverted!'
'''
    
    
    ##ssh -X ar6215@login.cx1.hpc.ic.ac.uk
"""
