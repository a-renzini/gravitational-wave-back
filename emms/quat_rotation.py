import qpoint as qp
import numpy as np
import healpy as hp
import pylab
import math
import cmath


def quat_mult(a,b):
    r = [0.,0.,0.,0.]
    r[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
    r[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
    r[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
    r[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
    return r

def quat_rot(q,p): #rot of p by q
    q_inv = [0.,0.,0.,0.]
    i = 0
    norm = q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]
    q_inv[i] = q[i]/norm
    i=1
    while i<4:
        q_inv[i] = -q[i]/norm
        i+=1
    return quat_mult(quat_mult(q,p),q_inv)

def quat_inv(q):
    norm = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
    q_inv = [0.,0.,0.,0.]
    q_inv[0] = q[0]/norm
    q_inv[1] = -q[1]/norm
    q_inv[2] = -q[2]/norm
    q_inv[3] = -q[3]/norm
    return q_inv