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
import math
import MapBack_pix as mb  #################
from matplotlib import cm
from mpi4py import MPI
ISMPI = True

#if mpi4py not present: ISMPI = False

import os
import sys

data_path = sys.argv[1]
out_path =  sys.argv[2]
maptyp = sys.argv[3]
noise_lvl = sys.argv[4]
noise_lvl = int(noise_lvl)
this_path = out_path

poi = False
if maptyp == 'planck_poi': poi = True

try:
    sys.argv[5]
except (NameError, IndexError):
    checkpoint = False
else:
    checkpoint = True
    checkfile_path = sys.argv[5]

    
if os.path.exists(data_path):
    print 'the data its in the ' , data_path
    # file exists                                                                                                             
if os.path.exists(out_path):
    print 'output goes to ' , out_path

#print 'the code its in the' , this_path
    
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
lmax = 2

##SIMULATION
sim = True

#INTEGRATING FREQS:                                                                                                           
low_f = 80.
high_f = 300.
low_cut = 80.
high_cut = 300.

    
#DETECTORS
dects = ['A','B','C', 'D', 'E', 'F']
ndet = len(dects)
nbase = int(ndet*(ndet-1)/2)
 
if myid == 0:
    
    if checkpoint == False and maptyp == 'gauss':
        
        map_in = mb.map_in_gauss(nside_in,noise_lvl)
        np.savez('%s/map_in%s.npz' % (this_path,noise_lvl), map_in = map_in )
        
        print '~~~~~~~~~~~~'
        print 'saved map_in_gauss in the out dir'
        print '~~~~~~~~~~~~'
    
    if checkpoint == True and maptyp == 'gauss':
        
        maptyp = 'checkfile'

#create object of class:
run = mb.Telescope(nside_in,nside_out,lmax, fs, low_f, high_f, dects, maptyp,this_path,noise_lvl)

if myid == 0:

    map_in = run.map_in
    
    if maptyp == 'planck':

        jet = cm.jet
        jet.set_under("w")
        hp.mollview(map_in,norm = 'hist', cmap = jet)
        plt.savefig('%s/map_in_%s.pdf' % (out_path,maptyp)  )
        plt.close('all')
    
    else:
        
        cbar = True
        if maptyp == '1pole': 
            cbar = False
            print 'the monopole is ',map_in[0]
            
        plt.figure()
        hp.mollview(map_in,cbar = cbar)
        plt.savefig('%s/map_in_%s.pdf' % (out_path,maptyp)  )
        plt.close('all')
        
                
else: map_in = None

map_in = comm.bcast(map_in, root=0)

# define start and stop time to search
# in GPS seconds
counter = 0

if checkpoint  == True:
    checkdata = np.load(checkfile_path)
    counter = checkdata['counter']

start = 1126224017  + np.int(60*counter) #1127000000 #O1 start GPS 1126051217 1126224017

#if checkpoint == True : start = start
        
stop  = 1129000000 #1137254417  #O1 end GPS     


###########################UNCOMMENT ME#########################################

print 'flagging the good data...'

if myid == 0:
    segs_begin, segs_end = run.flagger(start,stop,filelist)
    segs_begin = list(segs_begin)
    segs_end = list(segs_end)
    
    
    i = 0
    while i in np.arange(len(segs_begin)):
        delta = segs_end[i]-segs_begin[i]
        if delta > 15000:   #250 min
            steps = int(math.floor(delta/15000.))
            for j in np.arange(steps):
                step = segs_begin[i]+(j+1)*15000
                segs_end[i+j:i+j]=[step]
                segs_begin[i+j+1:i+j+1]=[step]
            i+=steps+1
        else: i+=1

    
else: 
    segs_begin = None
    segs_end = None
     
segs_begin = comm.bcast(segs_begin, root=0)
segs_end = comm.bcast(segs_end, root=0)


ctime_nproc = []
strain1_nproc = []
strain2_nproc = []
b_pixes = []


npix_out = hp.nside2npix(nside_out)

if myid == 0:
    Z_p = np.zeros(npix_out)
    S_p = np.zeros(npix_out)
    M_p_pp = 0.
    A_pp = 0.
    conds = []
    H1_PSD_fits = []
    L1_PSD_fits = []
    if checkpoint  == True:
        Z_p += checkdata['Z_p']
        M_p_pp += checkdata['M_p_pp']
        S_p = None
        A_pp += checkdata['A_pp'] 
        conds = checkdata['conds']
        print 'we are at minute', counter
    
    
else:
    Z_p = None
    S_p = None
    M_p_pp = None
    A_pp = None
    counter = 0

if checkpoint == True:
    map_in = comm.bcast(map_in, root=0)   

map_in_save = map_in.copy()

print 'segmenting the data...'


for sdx, (begin, end) in enumerate(zip(segs_begin,segs_end)):

    n=sdx+1

    
    
    if myid == 0:
        ctime, strain_H1, strain_L1 = run.segmenter(begin,end,filelist)
    
    else: 
        ctime = None
        strain_H1 = None
        strain_L1 = None
 
    ctime = comm.bcast(ctime, root=0)
    strain_H1 = comm.bcast(strain_H1, root=0)
    strain_L1 = comm.bcast(strain_L1, root=0)
    
    #strain_L1.highpass(10.)
    
    if len(ctime)<2 : continue
    
    idx_block = 0

    while idx_block < len(ctime):
        ctime_nproc.append(ctime[idx_block])
        strain1_nproc.append(strain_H1[idx_block])
        strain2_nproc.append(strain_L1[idx_block])
        
        
        if len(ctime_nproc) == nproc:   #################################
            
            #create empty lm objects:
        
            z_p = np.zeros(npix_out)
            my_A_pp = np.zeros(npix_out)
            my_M_p_pp = np.zeros((npix_out,npix_out))
            cond = 0.
            pix_bs_up = np.zeros(nbase)
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
            
            print 'filtering, ffting & saving the strains...'

            Nt = len(my_h1)
            Nt = lf.bestFFTlength(Nt)
            
            freqs = np.fft.rfftfreq(2*Nt, 1./fs)
            freqs = freqs[:Nt/2+1]
            
            mask = (freqs>low_f) & (freqs < high_f)
            
            strains = (my_h1,my_l1)
            strains_copy = (my_h1.copy(),my_l1.copy()) #calcualte psds from these

                    ###########################
            strains_f = []
            
            psds, flags = run.injector(strains_copy,my_ctime,low_cut,high_cut,poi)
            #strains are the new generated strains
            #
            
            avoid = False
            if sum(flags) > 0:
                avoid = True
                psds[0] = np.array([ 0.,   0.,   0.])
                psds[1] = np.copy(psds[0])
                
            if avoid is not True: 

                #print 'std of corr. t_stream: ', np.std(strains[0]*strains[1])
            
                psds_f = []
           
                for i in range(ndet):
                    psds_f.append(run.PDX(freqs,psds[i][0],psds[i][1],psds[i][2])*fs)#**2)
            
                if sim == False:
                    for i in range(ndet):
                        strains_f.append(run.filter(strains[i], low_cut,high_cut,psds[i])[mask])
                    
                    strains_f = [(strains_f[0]*np.conj(strains_f[1]))] #become correlated strains

            
            
            
                if sim == True:

                    print 'generating...'
                    h1_in = my_h1.copy()
                    l1_in = my_l1.copy()
                    strains_in = (h1_in,l1_in)
                    #print strains_in
                    strains_corr = run.injector(strains_in,my_ctime,low_cut,high_cut,poi, sim)[0]
                    strains_corr = run.noisy(strains_corr,psds_f,mask)
                
                
                    strains_f = strains_corr

            
                '''
                now strains_w, etc are pairs of 60s segments of signal, in frequency space.
                '''
            
                print 'filtering done'
                    #Integrate over frequency in the projector

                ####################################################################

                #proj_lm = np.array([np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)]*len(ctime)) #why *len_ctime?

                print 'running the projector, obtaining a dirty map'

                pix_bs = run.geometry(my_ctime)[0]
                pix_bs_up = run.geometry(my_ctime)[0]
                q_ns = run.geometry(my_ctime)[1]
                pix_ns = run.geometry(my_ctime)[2]
            
                #print pix_bs
            
                # fig = plt.figure()
                # hp.mollview(np.zeros_like(Z_p))
                # hp.visufunc.projscatter(hp.pix2ang(nside_in,pix_ns))
                # plt.savefig('n_pixs.pdf')
            
            
                # for i in range(len(pix_bs)):
                #     b_pixes.append(pix_bs[i])
            
                print 'time: ', my_ctime[0]
            
                z_p, my_M_p_pp, my_A_pp = run.projector(my_ctime,strains_f,psds_f,freqs,pix_bs, q_ns, norm = True)
                cond = np.linalg.cond(my_M_p_pp)
            
            if myid == 0:
                z_buffer = np.zeros_like(z_p)
                M_p_pp_buffer = np.zeros_like(my_M_p_pp)   
                A_pp_buffer = np.zeros_like(z_p)
                conds_array = np.zeros(nproc)
                a_buffer = nproc * [0.,0.,0.]
                pdx_H1 =  np.zeros_like(a_buffer)
                pdx_L1 =  np.zeros_like(a_buffer)
                b_buffer =  nproc * [nbase*[0]]
                
            else:
                z_buffer = None
                M_p_pp_buffer = None
                A_pp_buffer = None
                conds_array = None
                pdx_H1 = None
                pdx_L1 = None
                b_buffer = None
            
            if ISMPI:                 
                comm.barrier()
                comm.Reduce(z_p, z_buffer, root = 0, op = MPI.SUM)
                comm.Reduce(my_M_p_pp, M_p_pp_buffer, root = 0, op = MPI.SUM)
                comm.Reduce(my_A_pp, A_pp_buffer, root = 0, op = MPI.SUM)
                conds_array = comm.gather(cond, root = 0)
                pdx_H1 = comm.gather(psds[0],root = 0)
                pdx_L1 = comm.gather(psds[1], root = 0)
                b_buffer = comm.gather(pix_bs_up,root = 0)

                
                if myid ==0: counter += nproc

            else:
                z_buffer += z_p
                counter += 1
                M_p_pp_buffer += my_M_p_pp
                A_pp_buffer += my_A_pp
                conds.append(cond)


            if myid == 0:
                
                print 'this is id 0'
                Z_p += z_buffer
                M_p_pp += M_p_pp_buffer 
                A_pp += A_pp_buffer   
                
                conds_array = np.array(conds_array)
                np.append(conds,conds_array)
                H1_PSD_fits.append(pdx_H1)
                L1_PSD_fits.append(pdx_L1)
                b_pixes.append(b_buffer)
                
                H1_PSD_fits_flat = 0.
                L1_PSD_fits_flat = 0.
                
                H1_PSD_fits_flat = sum(H1_PSD_fits, [])
                L1_PSD_fits_flat = sum(L1_PSD_fits, [])
                
                b_pixes_flat = 0.
                b_pixes_flat = np.concatenate(b_pixes).ravel().tolist()
                np.savez('%s/b_pixes.npz' % out_path, b_pixes = b_pixes_flat )
                
                print '+++'
                print counter, 'mins analysed.'
                print '+++'

                #print 'M is ', len(M_lm_lpmp), ' by ', len(M_lm_lpmp[0])

                print 'Inverting M...'
                
                #### SVD

                M_p_pp_inv = np.linalg.pinv(M_p_pp,rcond=1.e-8)
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
                
                
                if counter % (nproc) == 0 or checkpoint == True:    ##
                    
                    # f = open('%s/M%s.txt' % (out_path,counter), 'w')
                    # print >>f, 'sim = ', sim
                    # print >>f, M_p_pp
                    # print >>f, '===='
                    # print >>f, M_p_pp_inv
                    # print >>f, '===='
                    # print >>f, np.linalg.eigh(M_p_pp)
                    # print >>f, '===='
                    # print >>f, cond
                    # print >>f, '===='
                    # print >>f, np.dot(M_p_pp, M_p_pp_inv),np.identity(len(M_p_pp))
                    # f.close()
                    
                    fits1 = 0.
                    fits2 = 0.
                    
                    fits1 = np.array(H1_PSD_fits_flat).T
                    fits2 = np.array(L1_PSD_fits_flat).T
                    fits1 = np.append(fits1,fits2,axis = 0) 
                    
                    # mark = 0
                    # new_fits = []
                    # while mark<counter:
                    #     for i in range(len(fits1)):
                    #         mark2 = mark+2*nproc
                    #         new_fits.append(fits1[i][mark:mark2])
                    #         mark+=2*nproc
                    #
                    # plt.matshow(new_fits)
                    # plt.colorbar()
                    # plt.savefig('%s/psdfits_mat.pdf' % out_path)
                    # plt.close('all')
                    
                    # plt.figure()
                    # plt.plot(fits1[0])
                    # plt.plot(fits1[1])
                    # plt.plot(fits1[2])
                    # plt.plot(fits1[3])
                    # plt.plot(fits1[4])
                    # plt.plot(fits1[5])
                    # plt.savefig('psdfits.pdf')               
                    
                    # fig = plt.figure()
                    # hp.mollview(Z_p)
                    # plt.savefig('%s/dirty_map%s.pdf' % (out_path, counter))


                    
                    hp.fitsfunc.write_map('%s/S_p%s.fits' % (out_path,counter), S_p*1.e30) 
                    
                    
                    # if maptyp == 'planck':
                    #
                    #     jet = cm.jet
                    #     jet.set_under("w")
                    #     hp.mollview(S_p,norm = 'hist', cmap = jet)
                    #     plt.savefig('%s/S_p%s.pdf' % (out_path,counter))
                    #     plt.close('all')
                    #
                    # else:
                    #     fig = plt.figure()
                    #     hp.mollview(S_p)
                    #     plt.savefig('%s/S_p%s.pdf' % (out_path,counter))
                    #     plt.close('all')
                    
                    np.savez('%s/checkfile.npz' % out_path, Z_p=Z_p, M_p_pp=M_p_pp, A_pp = A_pp,counter = counter, conds = conds, map_in = map_in_save )
                    
                    #if counter % (nproc) == 0:
                    np.savez('%s/checkfile%s.npz' % (out_path,counter), Z_p=Z_p, M_p_pp=M_p_pp,A_pp = A_pp, counter = counter, conds = conds, map_in = map_in_save )
                        
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
                
                    # fig = plt.figure()
                    # hp.mollview(np.zeros_like(Z_p))
                    # hp.visufunc.projscatter(hp.pix2ang(nside_in,b_pixes))
                    # plt.savefig('%s/b_pixs%s.pdf' % (out_path,counter))
                    
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
