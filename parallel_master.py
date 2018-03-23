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

nside_in = 16
nside_out = 16
lmax = 2
sim = True  
simtyp = 'mono'

#INTEGRATING FREQS:                                                                                                           
low_f = 80.
high_f = 300.
low_cut = 80.
high_cut = 300.

    
#DETECTORS
dects = ['H1','L1','V1']
ndet = len(dects)
nbase = int(ndet*(ndet-1)/2)
 
#create object of class:
run = mb.Telescope(nside_in,nside_out,lmax, fs, low_f, high_f, dects)


# define start and stop time to search
# in GPS seconds
start = 931035615 #S6 start GPS
stop  = 971622015  #S6 end GPS


###########################UNCOMMENT ME#########################################

print 'flagging the good data...'

segs_begin, segs_end = run.flagger(start,stop,filelist)

ctime_nproc = []
strain1_nproc = []
strain2_nproc = []
b_pixes = []

if myid == 0:
    Z_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
    S_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
    M_lm_lpmp = 0.
    M_lm_lpmp_2 = 0.
    counter = 0
    conds = []
    if checkpoint  == True:
        checkdata = np.load(checkfile_path)
        Z_lm += checkdata['Z_lm']
        M_lm_lpmp += checkdata['M_lm_lpmp']
        M_lm_lpmp_2 += checkdata['M_lm_lpmp_2']
        S_lm = None
        counter = checkdata['counter']
        conds = checkdata['conds']
        print counter
        
else:
    Z_lm = None
    S_lm = None
    M_lm_lpmp = None
    M_lm_lpmp_2 = None
    counter = 0

print 'segmenting the data...'

mlm = []

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
            m_lm = np.ones(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)
            my_M_lm_lpmp = 0.
            my_M_lm_lpmp_2 = 0.
            
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
                strains = run.injector(strains_in,my_ctime,low_cut,high_cut, sim,simtyp)[0]
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
                psds_f[i] = np.ones_like(psds_f[i])       ######weightless
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
            my_M_lm_lpmp_2 += run.M_lm_lpmp_t_2(my_ctime, psds_f,freqs,pix_bs,q_ns)
            
            print my_M_lm_lpmp-my_M_lm_lpmp_2
            
            cond = np.linalg.cond(my_M_lm_lpmp)

            if myid == 0:
                M_lm_lpmp_buffer = np.zeros_like(my_M_lm_lpmp)
                M_lm_lpmp_2_buffer = np.zeros_like(my_M_lm_lpmp)
                conds_array = np.zeros(nproc)
            else:
                M_lm_lpmp_buffer = None
                M_lm_lpmp_2_buffer = None
                conds_array = None

            if ISMPI:
                comm.barrier()
                comm.Reduce(my_M_lm_lpmp, M_lm_lpmp_buffer, root = 0, op = MPI.SUM)
                comm.Reduce(my_M_lm_lpmp_2, M_lm_lpmp_2_buffer, root = 0, op = MPI.SUM)
                comm.Gather(cond, conds_array, root = 0)

            else:
                M_lm_lpmp_buffer += my_M_lm_lpmp
                M_lm_lpmp_2_buffer += my_M_lm_lpmp_2
                conds.append(cond)

            if myid == 0:
                M_lm_lpmp += M_lm_lpmp_buffer
                M_lm_lpmp_2 += M_lm_lpmp_2_buffer                
                np.append(conds,conds_array)

                #print 'M is ', len(M_lm_lpmp), ' by ', len(M_lm_lpmp[0])

                print 'Inverting M...'

                #### SVD

                print M_lm_lpmp
                print M_lm_lpmp_2 
                M_inv = np.linalg.pinv(M_lm_lpmp)   #default:  for cond < 1E15
                M_inv_2 = np.linalg.pinv(M_lm_lpmp_2)   #default:  for cond < 1E15
                
                print M_inv, M_inv_2
                
                print 'the matrix has been inverted!'


                ################################################################

                S_lm = np.array(np.dot(M_inv,Z_lm)+np.dot(M_inv_2,np.conj(Z_lm))) #fully accumulated maps!
                
                print S_lm
                #S_lm+= s_lm

                #print len(s_lm)
                #print s_lm

                #dt_tot = np.sum(dt_lm,axis = 0)
                #print 'dt total:' , len(dt_tot.real)
                #print dt_tot
                
                if counter % (nproc) == 0:    ## *10000
                    
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
