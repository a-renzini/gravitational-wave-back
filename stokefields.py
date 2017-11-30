import qpoint as qp
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 4
lmax = nside

a_lm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)

#monopole (?)
a_lm[0] = 1.

cls = hp.sphtfunc.alm2cl(a_lm)

print cls 

# cls=[1]*nside
# i=0
# while i<nside:
#     cls[i]=1./(i+1.)**2.
#     i+=1

Istoke = hp.sphtfunc.alm2map(a_lm, nside)

#Istoke = np.vstack(hp.synfast(cls, nside=nside, pol=True, new=True)).flatten()
Qstoke = np.vstack(hp.synfast(cls, nside=nside, pol=True, new=True)).flatten()
Ustoke = np.vstack(hp.synfast(cls, nside=nside, pol=True, new=True)).flatten()
Vstoke = np.vstack(hp.synfast(cls, nside=nside, pol=True, new=True)).flatten()


fig = plt.figure()
hp.mollview(Istoke)
#hp.visufunc.projscatter(hp.pix2ang(nside,pix_bs))
#hp.visufunc.projscatter(hp.pix2ang(nside,pix_ns))
plt.savefig('Istoke.pdf' )

#test: monopoles, dipoles, with hp.alm2map