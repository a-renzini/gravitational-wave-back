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

# sampling rate:                                                                                                              
fs = 4096


nside_in = 16
nside_out = 8
lmax = 2
sim = True

#INTEGRATING FREQS:                                                                                                           
low_f = 80.
high_f = 300.
low_cut = 80.
high_cut = 300.

    
#DETECTORS
dects = ['H1','L1','V1']#,'A1']
ndet = len(dects)
nbase = int(ndet*(ndet-1)/2)
 
#create object of class:
run = mb.Telescope(nside_in,nside_out,lmax, fs, low_f, high_f, dects, 'gauss')

map_in = run.map_in

np.savez('map_in.npz', map_in = map_in )
