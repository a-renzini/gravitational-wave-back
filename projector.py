#need: rotation_pix()


        '''
        after segmenting the data and discarding the short segments, you feed the segments and relative ctimes
        into the dirty mapper (PROJECTION ROUTINE)
        '''

def dirty_map(ct_split, s_split, nside, lmax):    
            
    npix = hp.nside2npix(nside)
    
    # get quaternions for H1->L1 baseline (use as boresight)
    q_b = Q.azel2bore(np.degrees(az_b), np.degrees(el_b), None, None, np.degrees(H1_lon), np.degrees(H1_lat), ct_split)
    # get quaternions for bisector pointing (use as boresight)
    q_n = Q.azel2bore(0., 90.0, None, None, np.degrees(lonMid), np.degrees(latMid), ct_split)
    
    pix_b, s2p, c2p = Q.quat2pix(q_b, nside=nside, pol=True) #spin-2
    pix_n, s2p_n, c2p_n = Q.quat2pix(q_n, nside=nside, pol=True) #spin-2

    print '{}/{} {} seconds'.format(idx,n_split,len(ct_split)/fs)
    
    # Filter the data
    s_filt = lf.whitenbp_notch(s_split)

    # This is the 'projection' side of mapping equation
    # z_p = (A_tp)^T N_tt'^-1 d_t'= A_pt  N_tt'^-1 d_t'
    # It takes a timestream, inverse noise filters and projects onto
    # pixels p (here p are actually lm)
    # this is the 'dirty map'

    # The sky map is obtained by
    # s_p = (A_pt N_tt'^-1 A_t'p')^-1  z_p'
    
    #sum over time
    #for tidx, (p, s, quat) in enumerate(zip(pix_b,s_filt,q_n)):

    # average over sub segment and use
    # middle of segment for pointing etc.
    mid_idx = len(pix_n)/2
    p = pix_b[mid_idx]          
    quat = q_n[mid_idx]
    s = np.average(s_filt)

    # polar angles of baseline vector
    theta_b, phi_b = hp.pix2ang(nside,p)

    # rotate gammas
    # TODO: will need to oversample here
    # i.e. use nside > nside final map
    # TODO: pol gammas
    rot_m_array = rotation_pix(Q, np.arange(npix), quat) #rotating around the bisector of the gc 
    gammaI_rot = gammaI[rot_m_array]
    
    # Expand rotated gammas into lm
    glm = hp.map2alm(gammaI_rot, lmax, pol=False)

    # sum over lp, mp
    for l in range(lmax):
        for m in range(l):
            #print l, m
            idx_lm = hp.Alm.getidx(lmax,l,m)
            for lp in range(lmax):
                for mp in range(lp):
                    # remaining m index
                    mpp = m+mp
                    lmin_m = np.max([np.abs(l - lp), np.abs(m + mp)])
                    lmax_m = l + lp
                    for idxl, lpp in enumerate(range(lmin_m,lmax_m+1)):
                        data_lm[idx_lm] += ((-1)**mpp*(0+1.j)**lpp*fl[lpp]*sph_harm(mpp, lpp, theta_b, phi_b)*
                                            glm[hp.Alm.getidx(lmax,lp,mp)]*np.sqrt((2*l+1)*(2*lp+1)*(2*lpp+1)/4./np.pi)*
                                            threej_0[lpp,l,lp]*threej_m[lpp,l,lp,m,mp]*s)
                        hit_lm[idx_lm] += 1.
                        hits += 1.
    return hp.alm2map(data_lm/hits,nside,lmax=lmax)