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
import MapBack_pix as mb  #################
from mpi4py import MPI
ISMPI = True
#if mpi4py not present: ISMPI = False

import os
import sys

data_path = sys.argv[1]
out_path =  sys.argv[2]
maptyp = sys.argv[3]

try:
    sys.argv[4]
except (NameError, IndexError):
    checkpoint = None
else:
    checkpoint = True
    checkfile_path = sys.argv[4]

    
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

nside_in = 16
nside_out = 8
lmax = 2
sim = True  
simtyp = 'mono'

#INTEGRATING FREQS:                                                                                                           
low_f = 80.
high_f = 300.
low_cut = 80.
high_cut = 300.

    
#DETECTORS
dects = ['H1','L1']#,'V1']#,'A1']
ndet = len(dects)
nbase = int(ndet*(ndet-1)/2)
 
#create object of class:
run = mb.Telescope(nside_in,nside_out,lmax, fs, low_f, high_f, dects, maptyp)

map_in = run.get_map_in(maptyp)
        
plt.figure()
hp.mollview(map_in)
plt.savefig('%s/map_in.pdf' % out_path  )

# define start and stop time to search
# in GPS seconds
start = 931035615 #S6 start GPS
stop  = 971622015  #S6 end GPS      test 931135615 #


###########################UNCOMMENT ME#########################################

print 'flagging the good data...'

segs_begin, segs_end = run.flagger(start,stop,filelist)

ctime_nproc = []
strain1_nproc = []
strain2_nproc = []
b_pixes = []

npix_out = hp.nside2npix(nside_out)

if myid == 0:
    Z_p = np.zeros(npix_out)
    S_p = np.zeros(npix_out)
    M_p_pp = 0.
    counter = 0
    conds = []
    if checkpoint  == True:
        checkdata = np.load(checkfile_path)
        Z_p += checkdata['Z_p']
        M_p_pp += checkdata['M_p_pp']
        S_p = None
        counter = checkdata['counter']
        conds = checkdata['conds']
        map_in = checkdata['map_in']
        print counter
        
else:
    Z_p = None
    S_p = None
    M_p_pp = None
    counter = 0

print 'segmenting the data...'


for sdx, (begin, end) in enumerate(zip(segs_begin,segs_end)):

    n=sdx+1

    ctime, strain_H1, strain_L1 = run.segmenter(begin,end,filelist)

    if len(ctime)<2 : continue
    
    idx_block = 0

    while idx_block < len(ctime):
        ctime_nproc.append(ctime[idx_block])
        strain1_nproc.append(strain_H1[idx_block])
        strain2_nproc.append(strain_L1[idx_block])
        
        if len(ctime_nproc) == nproc:   #################################
                        #
            ######################################################
            ######################################################
            ######################################################

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


            #create empty lm objects:
            my_M_p_pp = 0.
            
            print 'filtering, ffting & saving the strains...'

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
                strains_corr = run.injector(strains_in,my_ctime,low_cut,high_cut, sim,simtyp)[0]
                
                
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
            
            psds_f = []

            for i in range(ndet):
                psds_f.append(psds[i](freqs)*fs**2) 
                #psds_f[i] = np.ones_like(psds_f[i])       ######weightless
                    
            
            #print strains_f[0][mask]*np.conj(strains_f[1])[mask]
            #print np.average(strains_f[0][mask]*np.conj(strains_f[1])[mask])
            
            print len(strains_corr), len(strains_corr[0]), len(psds_f), len(psds_f[0])
            
            '''
            now strains_w, etc are pairs of 60s segments of signal, in frequency space.
            '''

            print 'filtering done'
                #Integrate over frequency in the projector

            ####################################################################

            #proj_lm = np.array([np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)]*len(ctime)) #why *len_ctime?

            print 'running the projector, obtaining a dirty map'

            pix_bs = run.geometry(my_ctime)[0]
            q_ns = run.geometry(my_ctime)[1]
            pix_ns = run.geometry(my_ctime)[2]
            
            #print pix_bs
            
            # fig = plt.figure()
            # hp.mollview(np.zeros_like(Z_p))
            # hp.visufunc.projscatter(hp.pix2ang(nside_in,pix_ns))
            # plt.savefig('n_pixs.pdf')
            
            
            for i in range(len(pix_bs)):
                b_pixes.append(pix_bs[i])
            
            print 'time: ', my_ctime[0]
            
            z_p, my_M_p_pp = run.projector(my_ctime,strains_corr,psds_f,freqs,pix_bs, q_ns)
            cond = np.linalg.cond(my_M_p_pp)
            
            if myid == 0:
                z_buffer = np.zeros_like(z_p)
                M_p_pp_buffer = np.zeros_like(my_M_p_pp)   
                conds_array = np.zeros(nproc)
                
            else:
                z_buffer = None
                M_p_pp_buffer = None
                conds_array = None

            if ISMPI:
                comm.barrier()
                comm.Reduce(z_p, z_buffer, root = 0, op = MPI.SUM)
                comm.Reduce(my_M_p_pp, M_p_pp_buffer, root = 0, op = MPI.SUM)
                comm.Gather(cond, conds_array, root = 0)
                
                if myid ==0: counter += nproc

            else:
                z_buffer += z_p
                counter += 1
                M_p_pp_buffer += my_M_p_pp
                conds.append(cond)
            #print '----'
            #print 'z_lm', z_lm
            #print 'buffer', z_buffer
            #print 'counter',counter
            #print '----'


            if myid == 0:

                print 'this is id 0'
                Z_p += z_buffer
                M_p_pp += M_p_pp_buffer                
                np.append(conds,conds_array)
                
                print '+++'
                print counter, 'mins analysed.'
                print '+++'

                #print 'M is ', len(M_lm_lpmp), ' by ', len(M_lm_lpmp[0])

                print 'Inverting M...'
                
                #### SVD

                M_p_pp_inv = np.linalg.pinv(M_p_pp,rcond=1.e-5)
                print 'the matrix has been inverted!'
                
                S_p = np.einsum('...ik,...k->...i', M_p_pp_inv, Z_p)

                
                #fig = plt.figure()
                #hp.mollview(Z_p)
                #plt.savefig('%s/dirty_map%s.pdf' % (out_path, counter))

                #fig = plt.figure()
                #hp.mollview(S_p)
                #plt.savefig('%s/S_p%s.pdf' % (out_path,counter))
                
                
                
                ################################################################

                #S_p = np.array(np.dot(M_inv,Z_p)) #fully accumulated maps!

                #print len(s_lm)
                #print s_lm

                #dt_tot = np.sum(dt_lm,axis = 0)
                #print 'dt total:' , len(dt_tot.real)
                #print dt_tot
                
                if counter % (nproc*5) == 0:    ## *10000
                    
                    f = open('%s/M%s.txt' % (out_path,counter), 'w')
                    print >>f, 'sim = ', sim
                    print >>f, M_p_pp
                    print >>f, '===='
                    print >>f, M_p_pp_inv
                    print >>f, '===='                    
                    print >>f, np.linalg.eigh(M_p_pp)
                    print >>f, '===='
                    print >>f, cond
                    print >>f, '===='
                    print >>f, np.dot(M_p_pp, M_p_pp_inv),np.identity(len(M_p_pp))
                    f.close()
                    
                                        
                    
                    fig = plt.figure()
                    hp.mollview(Z_p)
                    plt.savefig('%s/dirty_map%s.pdf' % (out_path, counter))

                    fig = plt.figure()
                    hp.mollview(S_p)
                    plt.savefig('%s/S_p%s.pdf' % (out_path,counter))
                    
                    
                    np.savez('%s/checkfile%s.npz' % (out_path,counter), Z_p=Z_p, M_p_pp=M_p_pp, counter = counter, conds = conds, map_in = map_in )
                    
                    print 'saved dirty_map, clean_map and checkfile @ min', counter
                    
                    # falm = open('%s/alms%s.txt' % (out_path,counter), 'w')
                    # print >> falm, S_lm
                    # for l in range(lmax+1):
                    #     idxl0 =  hp.Alm.getidx(lmax,l,0)
                    #
                    #     almbit = 0.
                    #     for m in range(l+1):
                    #         idxlm =  hp.Alm.getidx(lmax,l,m)
                    #         almbit +=(2*S_lm[idxlm])*np.conj(S_lm[idxlm])/(2*l+1)
                    #
                    #     print >> falm, almbit - S_lm[idxl0]*np.conj(S_lm[idxl0])/(2*l+1)
                    #     print >> falm, np.average(S_p)
                    # print >> falm, 'end.'
                    # falm.close()
                
                    fig = plt.figure()
                    hp.mollview(np.zeros_like(Z_p))
                    hp.visufunc.projscatter(hp.pix2ang(nside_in,b_pixes))
                    plt.savefig('%s/b_pixs%s.pdf' % (out_path,counter))
                    
                    #exit()
                    
                    #if counter == 40:  
                ################################################    
                ################################################    
                ################################################    
            ctime_nproc = []
            strain1_nproc = []
            strain2_nproc = []
        
        #print idx_block    
        idx_block += 1
        #if idx_block == 1400: print S/N

if myid == 0:

    hp.mollview(Z_p)
    plt.savefig('%s/Z_p%s.pdf' % (out_path,counter))

    hp.mollview(S_p)
    plt.savefig('%s/S_p%s.pdf' % (out_path,counter))
    

    
    



    
    
    ##ssh -X ar6215@login.cx1.hpc.ic.ac.uk
