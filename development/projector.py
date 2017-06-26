def A_pt(self,ct_split):
        
    nside=self._nside
    lmax=self._lmax
        
    npix = self.npix
    
    # projector in matrix form: p = {lm} rows, t columns
    A_lm_t = np.array(len(ct_split)*[np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=complex)])
    hit_lm = np.zeros(len(A_lm_t[0]))

    # get quaternions for H1->L1 baseline (use as boresight)
    q_b = self.Q.azel2bore(np.degrees(self.az_b), np.degrees(self.el_b), None, None, np.degrees(self.H1_lon), np.degrees(self.H1_lat), ct_split)
    # get quaternions for bisector pointing (use as boresight)
    q_n = self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.lonMid), np.degrees(self.latMid), ct_split)

    pix_b, s2p, c2p = self.Q.quat2pix(q_b, nside=nside, pol=True) #spin-2
    pix_n, s2p_n, c2p_n = self.Q.quat2pix(q_n, nside=nside, pol=True) #spin-2

    # average over sub segment and use
    # middle of segment for pointing etc.
    mid_idx = len(pix_n)/2
    p = pix_b[mid_idx]          
    quat = q_n[mid_idx]

    # polar angles of baseline vector
    theta_b, phi_b = hp.pix2ang(nside,p)

    # rotate gammas
    rot_m_array = self.rotation_pix(np.arange(npix), quat) #rotating around the bisector of the gc 
    gammaI_rot = self.gammaI[rot_m_array]

    # Expand rotated gammas into lm
    glm = hp.map2alm(gammaI_rot, lmax, pol=False)
    fl = [self.freq_factor(l,0.01,300.) for l in range(lmax*4)]

    hits = 0.
    for i in range(ct_split):       # i = time index
        for l in range(lmax):
            for m in range(l):
                #print l, m
                idx_lm = hp.Alm.getidx(lmax,l,m)    # idx_lm = p index
                for lp in range(lmax):             # lp, mp, lpp, mpp = indices summed over to make A_lm
                    for mp in range(lp):
                        # remaining m index
                        mpp = m+mp
                        lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                        lmax_m = l + lp
                        for idxl, lpp in enumerate(range(lmin_m,lmax_m+1)):
                            A_lm_t[i][idx_lm] += ((-1)**mpp*(0+1.j)**lpp*fl[lpp]*sph_harm(mpp, lpp, theta_b, phi_b)*
                                                glm[hp.Alm.getidx(lmax,lp,mp)]*np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*
                                                self.threej_0[lpp,l,lp]*self.threej_m[lpp,l,lp,m,mp])
                            hit_lm[idx_lm] += 1.
                            hits += 1.