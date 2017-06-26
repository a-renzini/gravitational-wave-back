import numpy as np
import qpoint as qp
import healpy as hp
#import ligo_analyse_class as lac
from ligo_analyse_class_1 import Ligo_Analyse
import readligo as rl
import matplotlib.pyplot as plt


# sampling rate:
fs = 4096
ligo_data_dir = '/Users/pai/Data/'  
filelist = rl.FileList(directory=ligo_data_dir)

nside = 32
lmax = 16

run = Ligo_Analyse(nside,lmax)

# define start and stop time to search
# in GPS seconds
start = 931035615 #931079472 
stop  = 931622015 #931086336
#start = 931079472
#stop  = 931086336
#start = 931200000
#stop = 931300000


ctime, strain_x = run.segmenter(start,stop,filelist,fs)

out = []

for i in range(len(ctime)):
    out.append(run.projector(ctime[i], strain_x[i]))

hp.mollview(out[13])
plt.savefig('map13.pdf')

hp.mollview(np.sum(out, axis=0))
plt.savefig('map.pdf')
    
