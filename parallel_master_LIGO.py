import numpy as np
import qpoint as qp
import healpy as hp
import pylab
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

#FROM THE SHELL: data path, output path, type of input map, SNR level (noise =0, high, med, low)

data_path = sys.argv[1]
out_path =  sys.argv[2]
maptyp = sys.argv[3]
noise_lvl = sys.argv[4]
noise_lvl = int(noise_lvl)
this_path = out_path


# poisson masked "flickering" map

poi = False
if maptyp == 'planck_poi': poi = True


# if declared from shell, load checkpoint file 

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


# MPI setup for run 

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
####################################################################                                                          


# sampling rate; resolutions in/out
                                                                                                              
fs = 4096
nside_in = 32
nside_out = 8
npix_out = hp.nside2npix(nside_out)

# load the LIGO file list

ligo_data_dir = data_path                                                                         
filelist = rl.FileList(directory=ligo_data_dir)


# declare whether to simulate (correlated) data (in frequency space)
sim = False


# frequency cuts (integrate over this range)
                                                                                                          
low_f = 80.
high_f = 300.

    
# DETECTORS (should make this external input)

dects = ['H1','L1']
ndet = len(dects)
nbase = int(ndet*(ndet-1)/2)
 

# GAUSSIAN SIM. INPUT MAP CASE: make sure that the background map isn't re-simulated between scans, 
# and between checkfiles 
 
if myid == 0:
    
    if checkpoint == False and maptyp == 'gauss':
        
        map_in = mb.map_in_gauss(nside_in,noise_lvl)
        np.savez('%s/map_in%s.npz' % (this_path,noise_lvl), map_in = map_in )
        
        print '~~~~~~~~~~~~'
        print 'saved map_in_gauss in the out dir'
        print '~~~~~~~~~~~~'
    
    if checkpoint == True and maptyp == 'gauss':
        
        maptyp = 'checkfile'
        
        # when maptyp is checkfile it knows it doesn't need to re-make it, just pick it up from checkfile


# INITIALISE THE CLASS  ######################
# args of class: nsides in/out; sampling frequency; freq cuts; declared detectors; the path of the checkfile; SNR level

run = mb.Telescope(nside_in,nside_out, fs, low_f, high_f, dects, maptyp,this_path,noise_lvl)

##############################################


# PARALLELISATION : ID = 0 keeps track of work and stores the input maps, operators, etc.

 # create the input map (or pick it up if maptyp = checkfile) and broadcast it to ID neq 0

if myid == 0:
    
    map_in = run.map_in
    
    
    #save a plot of the input map (can remove this/make it optional)
    
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

##########################  RUN  TIMES  #########################################

# RUN TIMES : define start and stop time to search in GPS seconds; 
# if checkpoint = True make sure to start from end of checkpoint

counter = 0         #counter = number of mins analysed
start = 1126224017  #start = start time of O1 ...

if checkpoint  == True:
    checkdata = np.load(checkfile_path)
    counter = checkdata['counter'] 
    start = np.int(checkdata['checkstart'])  # ... start = checkpointed endtime
        
stop  = 1137254417  #O1 end GPS     


##########################################################################

########################### data  massage  #################################

# FLAGGING; SEGMENTING

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
    
# broadcast the segments heads+tails to id neq 0    
     
segs_begin = comm.bcast(segs_begin, root=0)
segs_end = comm.bcast(segs_end, root=0)

##############################  SETUP  OF  THE  RUN  #################################
# create empty objects for every processor: 
#ctime array; LH strain array; LL strain array; baseline pixels array (1 item pm)

ctime_nproc = []
strain1_nproc = []
strain2_nproc = []
b_pixes = []

# create empty objects for just ID = 0: 

# Z_p total dirty map (summed over the minutes and the baselines)
# S_p total clean map (re-obtained perdiocally from M_p_pp^-1 * Z_p => updated)
# M_p_pp total beam-pattern matrix (summed over the minutes and the baselines)
# A_pp total norm matrix: beam-pattern w/out NOISE (summed over the minutes and the baselines) ## -> NORMALI
# A_p total projector: dirty map w/out DATA & NOISE (summed over the minutes and the baselines) ## -> SATION

# conds condition number array (1 item pm - continuously updated)
# H1_PSD_fits / L1_PSD_fits sets of 3 fit parameters to LIGO PSDs: accumulated 1 pm with format array([a,b,c]) 

# objects above are read from checkfile if checkpoint = True; ESSENTIAL AS OBJECTS ARE ACCUMULATED OVER TIME 

if myid == 0:
    Z_p = np.zeros(npix_out)
    S_p = np.zeros(npix_out)
    M_p_pp = 0.
    A_pp = 0.
    A_p = 0.
    conds = []
    endtime = 0
    H1_PSD_fits = []
    L1_PSD_fits = []
    if checkpoint  == True:
        Z_p += checkdata['Z_p']
        M_p_pp += checkdata['M_p_pp']
        S_p = None                          # final clean map gets re-estimated every time
        A_p += checkdata['A_p'] 
        A_pp += checkdata['A_pp'] 
        conds = checkdata['conds']          # keep appending to conds array
        print 'we are at minute', counter , 'with startime' , start   
   
# (objs are empty for ID neq 0 ) 

else:
    Z_p = None
    S_p = None
    M_p_pp = None
    A_p = None
    A_pp = None
    counter = 0

# broadcast checkpointed input map to every proc 

if checkpoint == True:
    map_in = comm.bcast(map_in, root=0)   

# save a copy of the map for the checkfile; this is a safety fool-proof measure

map_in_save = map_in.copy()


########################### data  massage 2  #################################
# SEGMENTING THE DATA & HANDING IT OUT
# this is done efficiently : number of segments handed out per iteration of algorithm = nproc

print 'segmenting the data...'


for sdx, (begin, end) in enumerate(zip(segs_begin,segs_end)):

    n=sdx+1

    # ID = 0 segments the data

    if myid == 0:
        ctime, strain_H1, strain_L1 = run.segmenter(begin,end,filelist)
    
    else: 
        ctime = None
        strain_H1 = None
        strain_L1 = None
 
    # then each ID neq zero gets a copy
    
    ctime = comm.bcast(ctime, root=0)
    strain_H1 = comm.bcast(strain_H1, root=0)
    strain_L1 = comm.bcast(strain_L1, root=0)
    
    
    if len(ctime)<2 : continue      #discard short segments (may up this to ~10 mins)
    
    
    #idx_block: keep track of how many mins we're handing out
    
    idx_block = 0

    while idx_block < len(ctime):
        
        # accumulate ctime, strain arrays of length exactly nproc 
        
        ctime_nproc.append(ctime[idx_block])
        strain1_nproc.append(strain_H1[idx_block])
        strain2_nproc.append(strain_L1[idx_block])
        
        
        if len(ctime_nproc) == nproc:   # when you hit nproc start itearation
            
            # create personal proc empty objects:
            
            # z_p personal dirty map (summed over the baselines)
            # my_M_p_pp personal beam-pattern matrix (summed over the baselines)
            # my_A_pp personal norm matrix: beam-pattern w/out NOISE (summed over the baselines) 
            # my_A_p personal projector: dirty map w/out DATA & NOISE (summed over the baselines)

            # cond condition number array (1 item pm -> will be accumulated in chuncks of nproc)
            
            z_p = np.zeros(npix_out)
            my_A_p = np.zeros(npix_out)
            my_A_pp = np.zeros((npix_out,npix_out))
            my_M_p_pp = np.zeros((npix_out,npix_out))
            cond = 0.
            pix_bs_up = np.zeros(nbase)


            # hand out the work to the procs
            
            idx_list = np.arange(nproc)

            if myid == 0:
                my_idx = np.split(idx_list, nproc)  
                
            else:
                my_idx = None


            if ISMPI:
                my_idx = comm.scatter(my_idx)
                my_ctime = ctime_nproc[my_idx[0]]
                my_h1 = strain1_nproc[my_idx[0]]
                my_l1 = strain2_nproc[my_idx[0]]
                my_endtime = my_ctime[-1]
            
            
            
            ########################### data  massage 3  #################################
            # FILTERING/SIMULATING & FFTing THE DATA; PREPPING IT FOR  MAPPING
            
            print 'filtering, ffting & saving the strains...'
            
            # Fourier space objects: Nt optimal timestream length; freqs frequency array at chosen fs
            
            Nt = len(my_h1)
            Nt = lf.bestFFTlength(Nt)
            
            freqs = np.fft.rfftfreq(2*Nt, 1./fs)
            freqs = freqs[:Nt/2+1]
            
            
            # frequency mask
            
            mask = (freqs>low_f) & (freqs < high_f)
            
            
            # repackage the strains & copy them (fool-proof); create empty array for the filtered, FFTed, correlated data
            
            strains = (my_h1,my_l1)
            strains_copy = (my_h1.copy(),my_l1.copy()) #calcualte psds from these
            
            strains_f = []
            
            
            # HERE WE GO: at this stage, we use injector() to recover the psd params & flags (needed to discard poor-fit segs)
            
            psds, flags = run.injector(strains_copy,my_ctime,low_f,high_f,poi)
            #
            # psds shape: [array([a1,b1,c1]), array([a2,b2,c2])]
            #
            
            # if there is a flagged minute, we discard it 
            
            avoid = False
            if sum(flags) > 0:
                avoid = True
                psds[0] = np.array([ 0.,   0.,   0.])
                psds[1] = np.copy(psds[0])
            
                
            if avoid is not True: 
                
                # create empty array for the psds: these will just be the function PDX from the class with the
                # params [a,b,c] estimated above sampled at freqs. LENGTH = NUMBER OF DECTS
                
                psds_f = []
           
                for i in range(ndet):
                    psds_f.append(run.PDX(freqs,psds[i][0],psds[i][1],psds[i][2]))
            
                
                # FORK: if using real data, run filter() on it. If simulating, create sim corr data stream.
                # Fill strains_f created above.  
                
                if sim == False:
                    
                    for i in range(ndet):
                        
                        strains_f.append(run.filter(strains[i], low_f,high_f,psds[i])[mask])
                
                    strains_f = [(strains_f[0]*np.conj(strains_f[1]))] # become correlated strains
            
    
                if sim == True:

                    print 'generating...'
                    
                    h1_in = my_h1.copy()
                    l1_in = my_l1.copy()
                    strains_in = (h1_in,l1_in)
                    
                    strains_corr = run.injector(strains_in,my_ctime,low_f,high_f,poi, sim)[0]
                    strains_corr = run.noisy(strains_corr,psds_f,mask)
                
                    strains_f = strains_corr
            
                print 'filtering done'

                ################################################################################
                ########################### data  massage over  ################################
                # NOW THE GOOD STUFF
                
                print 'running the projector, obtaining a dirty map'

                # PREP: run geometry() to get, for each minute:
                # pix_bs - the pixels the baselines are pointing at
                # q_ns - the quaternions of the zenith of the detectors
                # pix_ns - the pixels corresponding to q_ns
                
                # BONUS: pix_bs_up - a much higher resolution b pixel for trace-plot making 
                
                pix_bs = run.geometry(my_ctime)[0]
                pix_bs_up = run.geometry_up(my_ctime)[0]
                q_ns = run.geometry(my_ctime)[1]
                pix_ns = run.geometry(my_ctime)[2]
            
                
                ### NEW : KEEPING TRACK OF TIME
                # print the start time and save the end time of each segment; will select the max_endtime
                # to hand down to the checkfile
                
                print 'time: ', my_ctime[0]
                
                #my_endtime = my_ctime[-1]
            
                # THIS IS IT: apply the projector() to the correlated data
                # saving:
                # z_p which was personal dirty map, summed over the baseline
                # my_M_p_pp personal beam-pattern matrix
                # my_A_p
                # my_A_pp
                
                # condition number for the beam-pattern
                
                z_p, my_M_p_pp, my_A_p, my_A_pp = run.projector(my_ctime,strains_f,psds_f,freqs,pix_bs, q_ns, norm = True)
                cond = np.linalg.cond(my_M_p_pp)
            
            # out of the loop: each proc has a personal set of dirty maps and beam-patterns
            # create buffers now to accumulate these operators
            
            # BUFFERS:
            # z_buffer dirty map
            # A_p_buffer, A_pp_buffer for the norms
            # M_p_pp_buffer beam-pattern
            # conds_array to save the condition numbers as we go  (maybe should calculate the condition at the very end - 
            # this may explain fuzziness of conds )
            # a_buffer is just a format;
            # pdx_H1/L1 params of the PDX fit
            # b_buffer collection of baseline pixels
            
            if myid == 0:
                
                z_buffer = np.zeros_like(z_p)
                A_p_buffer = np.zeros_like(z_p)
                M_p_pp_buffer = np.zeros_like(my_M_p_pp)   
                A_pp_buffer = np.zeros_like(my_M_p_pp)
                conds_array = np.zeros(nproc)
                endtimes_array = np.zeros(nproc)
                a_buffer = nproc * [0.,0.,0.]
                pdx_H1 =  np.zeros_like(a_buffer)
                pdx_L1 =  np.zeros_like(a_buffer)
                b_buffer =  nproc * [nbase*[0]]
                
                
            else:
                
                z_buffer = None
                M_p_pp_buffer = None
                A_pp_buffer = None
                A_p_buffer = None
                conds_array = None
                endtimes_array = None
                pdx_H1 = None
                pdx_L1 = None
                b_buffer = None
            
            
            # let's collect the winnings: Reduce sums over the od 0 dimension, gather returns a list
            # NOW THE BUFFERS ARE THE SUM OVER TIME OF THE DIRTY MAP/BEAM-PATTERN OVER nproc MINUTES
            
            if ISMPI: 
                                
                comm.barrier()
                comm.Reduce(z_p, z_buffer, root = 0, op = MPI.SUM)
                comm.Reduce(my_M_p_pp, M_p_pp_buffer, root = 0, op = MPI.SUM)
                comm.Reduce(my_A_pp, A_pp_buffer, root = 0, op = MPI.SUM)
                comm.Reduce(my_A_p, A_p_buffer, root = 0, op = MPI.SUM)
                conds_array = comm.gather(cond, root = 0)
                endtimes_array = comm.gather(my_endtime, root = 0)
                pdx_H1 = comm.gather(psds[0],root = 0)
                pdx_L1 = comm.gather(psds[1], root = 0)
                b_buffer = comm.gather(pix_bs_up,root = 0) # saving the high res b_pixes to use in plots 
                
                
                if myid ==0: 
                    
                    counter += nproc
                    endtime = max(endtimes_array)
                    
                    from astropy.time import Time
                    t = Time(endtime, format='unix')
                    t = np.int(Time(t, format='gps').value)
                    
                    endtime = t

                    
            else:
                z_buffer += z_p
                counter += 1
                M_p_pp_buffer += my_M_p_pp
                A_pp_buffer += my_A_pp
                A_p_buffer += my_A_p
                conds.append(cond)
                endtime = my_endtime

                
            # LAST STEPS: 
            # update the dirty map & beam-pattern
            # repackage arrays for checkpointing
            # invert the beam-pattern and save a clean map to look at
                
            if myid == 0:
                
                print 'this is id 0'
                
                Z_p += z_buffer
                M_p_pp += M_p_pp_buffer 
                A_p += A_p_buffer 
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
                np.savez('%s/b_pixes.npz' % out_path, b_pixes = b_pixes_flat ) #save the b_pixes file 
                
                print '+++'
                print counter, 'mins analysed.'
                print '+++'

                print 'Inverting M...'
                
                #### SVD invert the beam-pattern

                M_p_pp_inv = np.linalg.pinv(M_p_pp,rcond=1.e-8)
                print 'the matrix has been inverted!'
                
                S_p = np.einsum('...ik,...k->...i', M_p_pp_inv, Z_p)
                
                
                ################################################################
                #
                # only checkpoint once in a while - set step to custom
                #
                
                step = 1
                
                if counter % (nproc*step) == 0 or checkpoint == True:    
                    
                    fits1 = 0.
                    fits2 = 0.
                    
                    fits1 = np.array(H1_PSD_fits_flat).T
                    fits2 = np.array(L1_PSD_fits_flat).T
                    fits1 = np.append(fits1,fits2,axis = 0) 

                    # can save the list of fits params if so wish
                    
                    # save a fits file for the clean map
                    # !!! NOTE !!! need to *1.e30 otherwise the numbers are to small (unsure why)
                    
                    hp.fitsfunc.write_map('%s/S_p%s.fits' % (out_path,counter), S_p*1.e30) 
                    
                    # save checkfile with
                    # Z_p accumulated dirty map
                    # M_p_pp    "     beam-pattern
                    # A_p , A_pp "    norms 
                    # counter number of minutes analysed
                    # checkstart is the max endtime of the run - will be the new start of the next one
                    # conds progressive conditions on M_p_pp 
                    # map_in input map for the simulated data 
                    
                    np.savez('%s/checkfile.npz' % out_path, Z_p=Z_p, M_p_pp=M_p_pp, A_p = A_p, A_pp = A_pp,counter = counter, checkstart = endtime, conds = conds, map_in = map_in_save )
                    
                    #if counter % (nproc) == 0:
                    np.savez('%s/checkfile%s.npz' % (out_path,counter), Z_p=Z_p, M_p_pp=M_p_pp, A_p = A_p, A_pp = A_pp, counter = counter, checkstart = endtime, conds = conds, map_in = map_in_save )
                        
                    print 'saved dirty_map, clean_map and checkfile @ min', counter, 'with endtime', endtime
                  
                    
            #empty the lists to refill with other nproc segments
                
            ctime_nproc = []
            strain1_nproc = []
            strain2_nproc = []
        

        idx_block += 1


# FINALE: WHEN WE REACH THE END OF THE RUN 

if myid == 0:

    hp.mollview(Z_p)
    plt.savefig('%s/Z_p%s.pdf' % (out_path,counter))

    hp.mollview(S_p)
    plt.savefig('%s/S_p%s.pdf' % (out_path,counter))
    

    
    



    
    
    ##ssh -X ar6215@login.cx1.hpc.ic.ac.uk
