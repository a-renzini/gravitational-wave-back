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
import BigClass as bc
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


nside = 8
lmax = 2
sim = True

#INTEGRATING FREQS:                                                                                                           
low_f = 80.
high_f = 300.
low_cut = 80.
high_cut = 300.

    
#DETECTORS
dects = ['H1','L1']#,'V1']
ndet = len (dects)
nbase = int(ndet*(ndet-1)/2)
 
#create object of class:
run = bc.Telescope(nside,lmax, fs, low_f, high_f)

# define start and stop time to search
# in GPS seconds
start = 931079472    #931079472: 31 segs   931158100: 69 segs  931168100: 7 segs
stop  = 931622015 #971622015 #931622015 #931086336 #


###########################UNCOMMENT ME#########################################

# print 'flagging the good data...'
#
# segs_begin, segs_end = run.flagger(start,stop,filelist)
#
# ctime_tot = []
# strain_1_tot = []
# strain_2_tot = []
#
# print 'segmenting the data...'
#
# for sdx, (begin, end) in enumerate(zip(segs_begin,segs_end)):
#
#     n=sdx+1
#
#     ctime, strain_H1, strain_L1 = run.segmenter(begin,end,filelist)
#
#     if len(ctime)<2 :
#         print 'too short!'
#         continue
#
#     ctime_tot.append(ctime)
#     strain_1_tot.append(strain_H1)
#     strain_2_tot.append(strain_L1)
#
# ctime_flat = []
# strain_1_flat = []
# strain_2_flat = []
#
# for a in range(len(ctime_tot)):
#     for b in range(len(ctime_tot[a])):
#         ctime_flat.append(ctime_tot[a][b])
#         strain_1_flat.append(strain_1_tot[a][b])
#         strain_2_flat.append(strain_2_tot[a][b])
#
#
# np.savez('%s/datafile_tot.npz' % (out_path), ctime_flat = ctime_flat, strain_1_flat = strain_1_flat, strain_2_flat = strain_2_flat )

####################################################################

flat_data = np.load('%s/datafile_tot.npz' % (out_path))

ctime_flat = flat_data['ctime_flat']
#strain_1_flat = flat_data['strain_1_flat']
#strain_2_flat = flat_data['strain_2_flat']

#print len(segs_begin)

idx_list = np.arange(len(ctime_flat))

if myid == 0: 
    my_idx = np.split(idx_list, nproc)
    
else:
    my_idx = None


if ISMPI:
        
    my_idx = comm.scatter(my_idx)


my_ctime_flat = ctime_flat[my_idx]

if myid == 3:
    print myid, my_ctime_flat[0], my_ctime_flat[-1]

exit()

############## EOF
############## EOF
############## EOF
############## EOF
############## EOF
############## EOF
############## EOF
############## EOF
############## EOF
############## EOF
############## EOF



my_strain_1_flat = None
my_strain_2_flat = None

#create empty lm objects:
m_lm = np.ones(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)

my_M_lm_lpmp = 0.
    
if myid == 0:
    Z_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
    S_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
    M_lm_lpmp = 0.
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
    Z_lm = None
    S_lm = None
    M_lm_lpmp = None
    counter = 0

for sdx, (ctime, h1, l1) in enumerate(zip(my_ctime_flat,my_strain_1_flat,my_strain_2_flat)):
    
    print sdx
    
    exit()

    ####################################################################

    strains_split = []
    psds_split = []

    print 'filtering, ffting & saving the strains...'

    Nt = len(strain_H1[0])
    Nt = lf.bestFFTlength(Nt)
    
    for idx_str, (h1, l1) in enumerate(zip(strain_H1, strain_L1)):
        #print idx_str        
        # FT
        
        # plt.figure()
        # plt.plot(h1, color = 'r')
        # plt.plot(l1, color = 'b')
        # #plt.ylim([-100.,100.])
        # plt.savefig('testa.png' )
        
        #datah1 = 'datah1'
        #np.savez(datah1,h1)
        
        freqs = np.fft.rfftfreq(2*Nt, 1./fs)
        freqs = freqs[:Nt/2+1]
        
        mask = (freqs>low_f) & (freqs < high_f)

        
        # use a copy of the strains so that the filtering works smoothly
        strains = (h1,l1)
        strains_copy = (h1.copy(),l1.copy()) #calcualte psds from these
        
        ###########################
        
        
        if sim == True:
            h1_in = h1.copy()
            l1_in = l1.copy()
            strains_in = (h1_in,l1_in)
            strains = run.injector(strains_in,low_cut,high_cut, sim)[0]
            

            #plt.figure()
            #plt.plot(freqs[mask],strains[1][mask])
            #plt.savefig('fakestream.pdf')

        #pass the noisy strains to injector got the psds
        psds = run.injector(strains_copy,low_cut,high_cut)[1]
        
        # plt.figure()
        # plt.loglog(freqs,psds[1](freqs)*fs**2)
        # plt.savefig('%s/fakestream_psd_pre.png' % out_path)
        #
        # plt.figure()
        # plt.plot(strains[1])
        # plt.savefig('%s/fakestream_1_pre.png' % out_path)
        #
        # plt.figure()
        # plt.loglog(freqs,np.abs(np.fft.rfft(strains[1], Nt)))
        # plt.savefig('%s/fakestream_f_pre.png' % out_path)

        
        strains_f = []
        psds_f = []
        strains_w = []
        
        for i in range(ndet):
            strains_f.append(run.filter(strains[i], low_cut,high_cut,psds[i]))
            psds_f.append(psds[i](freqs)*fs**2 )
            strains_w.append(strains_f[i]/(psds_f[i]))
        
        strains_split.append(strains_w)
        psds_split.append(psds_f)

        #plt.figure()
        # plt.loglog(freqs[mask],psds_f[1][mask])
        # plt.savefig('fakestream_psd.png')
        #
        # plt.figure()
        # plt.loglog(freqs[mask],np.abs(strains_f[1][mask]))
        # plt.savefig('fakestream_f1.png')
        #
        # plt.figure()
        # plt.plot(freqs[mask],np.abs(strains_f[1][mask]/psds_f[1][mask] ))
        # plt.savefig('fakestream_white.pdf')
        #
        # exit()
        
        
        '''
        now strain_x is a segment of 60 seconds of correlated signal, in frequency space.
        '''

    
    #print '+++'
    #print run.sim_tstream(ctime[0],1.,1.,freqs)
    print 'filtering done'
    #Integrate over frequency in the projector
        
    ####################################################################
        
    #proj_lm = np.array([np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)]*len(ctime)) #why *len_ctime?

    print 'running the projector, obtaining a dirty map'
    
    pix_bs = []
    q_ns = []
    pix_ns = []
    
    for i in range(len(ctime)):
        pix_bs.append(run.geometry(ctime[i])[0])
        q_ns.append(run.geometry(ctime[i])[1])
        pix_ns.append(run.geometry(ctime[i])[2])
    

    
    z_lm = run.projector(ctime,strains_split,freqs,pix_bs, q_ns)
    
    if myid == 0:
        z_buffer = np.zeros_like(z_lm)
        y = np.asarray(0)
    else: 
        z_buffer = None
        y = None
    
    count = np.asarray(count) 
    
    if ISMPI:
        comm.barrier()
        comm.Reduce(z_lm, z_buffer, root = 0, op = MPI.SUM)
        comm.Reduce(count, y , root = 0, op = MPI.SUM)
    
    else: 
        z_buffer += z_lm
        y += count
    #print '----'
    #print 'z_lm', z_lm
    #print 'buffer', z_buffer
    #print 'counter',counter
    #print '----'


    if myid == 0: 
        
        print 'this is id 0'
        Z_lm += z_buffer
        counter += y 
        print '+++'
        print counter, 'mins analysed.'
        print '+++'

        
        #Z_lm +=z_lm
    
    
        # for idx_t, ct_split in enumerate(ctime):
        #     ones = [1.]*len(freqs)
        #     proj_lm[idx_t] = run.summer(ctime[idx_t],ones,freqs) 
    
        #dirty_map_lm = hp.alm2map(np.sum(dt_lm,axis = 0),nside,lmax=lmax)
        dirty_map = hp.alm2map(Z_lm,nside,lmax=lmax)
    
        fig = plt.figure()
        hp.mollview(dirty_map)
        #hp.visufunc.projscatter(hp.pix2ang(nside,pix_bs))
        #hp.visufunc.projscatter(hp.pix2ang(nside,pix_ns))
        plt.savefig('%s/dirty_map%s.pdf' % (out_path, sdx))
        #dirty_map_lm = hp.alm2map(np.sum(dt_lm,axis = 0),nside,lmax=lmax)

        print 'saved: dirty_map.pdf'
    
        
        ####################################################################
    
        print 'building M^-1:'
    
    for idx_t in range(len(ctime)):
        print idx_t
        my_M_lm_lpmp += run.M_lm_lpmp_t(ctime[idx_t], psds_split[idx_t],freqs,pix_bs[idx_t],q_ns[idx_t])
    
    cond = np.linalg.cond(my_M_lm_lpmp)

    if myid == 0:
        M_lm_lpmp_buffer = np.zeros_like(my_M_lm_lpmp)
        conds_array = []
    else: 
        M_lm_lpmp_buffer = None
        conds_array = None
    
    if ISMPI:
        comm.barrier()
        comm.Reduce(my_M_lm_lpmp, M_lm_lpmp_buffer, root = 0, op = MPI.SUM)
        comm.gather(cond, conds_array , root = 0)

    else: 
        M_lm_lpmp_buffer += my_M_lm_lpmp
        conds.append(cond)

    if myid == 0: 
        M_lm_lpmp += np.real(M_lm_lpmp_buffer)
        conds.append(conds_array)

        #print 'M is ', len(M_lm_lpmp), ' by ', len(M_lm_lpmp[0])
    
        
        print 'Inverting...'
        
        #### SVD
        
        M_lm_lpmp = np.real(M_lm_lpmp)
        M_inv = np.linalg.pinv(M_lm_lpmp)   #default:  for cond < 1E15 
        
        print 'the matrix has been inverted!'
        
        f = open('%s/M%s.txt' % (out_path,sdx), 'w')
        print >>f, M_lm_lpmp
        print >>f, '===='
        print >>f, np.linalg.eigh(M_lm_lpmp)
        print >>f, '===='
        print >>f, cond
        print >>f, '===='
        print >>f, np.allclose(np.dot(M_lm_lpmp,M_inv),np.identity(len(M_lm_lpmp)))
        f.close
    
        ################################################################

        #s_lm = []
        S_p = []
        
        S_lm = np.array(np.dot(M_inv,Z_lm)) #fully accumulated maps!
        #S_lm+= s_lm
        
        S_p = hp.alm2map(S_lm,nside,lmax=lmax)
        #print len(s_lm)
        #print s_lm
    
        #dt_tot = np.sum(dt_lm,axis = 0)
        #print 'dt total:' , len(dt_tot.real)
        #print dt_tot

        hp.mollview(S_p)
        plt.savefig('%s/S_p%s.pdf' % (out_path,sdx))
        
        np.savez('%s/checkfile%s.npz' % (out_path,sdx), sdx=sdx, Z_lm=Z_lm, M_lm_lpmp=M_lm_lpmp, counter = counter, conds = conds )
    

if myid == 0:

    hp.mollview(hp.alm2map(Z_lm/len(ctime),nside,lmax=lmax))
    plt.savefig('%sZ_p%s.pdf' % (out_path,sdx))

    hp.mollview(hp.alm2map(S_lm/len(ctime),nside,lmax=lmax))
    plt.savefig('%sS_p%s.pdf' % (out_path,sdx))
    

    
exit()
    
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
