
import numpy as np

#no_O1 = np.array([ 34.70, 35.30,35.90, 36.70, 37.30, 40.95, 60.00, 120.00, 179.99, 304.99, 331.9, 499.0, 500.0, 510.02,  1009.99])
no_O2 = np.array([30.25, 31.25,32.25,33.0,34.5,35.25,36.25,37.0,40.5,41.75,45.5,46.0,59.6,299.5,305.0,315.4,331.5,500.25])

#sig_O1 = np.array([.5,.5,.5,.5,.5,.5,.5,1.,1.,1.,1.,2.,2.,2.,1.])
sig_O2 = np.array([.02,.02,.02,.02,.02,.02,.02,0.1,.01,.2,.2,.2,.2,1.,1.,.2,.1,8.])



notches_O1 = np.array([
[34.7,0.5],
[35.3,0.5],
[35.9,.5],
[36.7,.5],      #calibration H
[37.3,.5],      #calibration H
[40.95,.5],
[60,.5],
[120.,1.],
[179.99,1.],
[299.6,1.],      #beam splitter violin H
[302.22,1.],    #beam splitter violin H
[303.31,1.],    #beam splitter violin H
[306.99,1.],    #beam splitter violin L
[307.34,1.],    #beam splitter violin L
[307.5,1.],    #beam splitter violin L
[315.1,1.],    #beam splitter violin L
[331.9,1.],     #calibration H
[333.33,1.],     #calibration H
[499.0,2.],
[500.,2.],
[510.02,2.],
[47.69,.1],
[44.69,.1],     #coherent with PEM channel(s)
[100.,.02],
[453.,0.05]])

'''
COMBS
'''

comb_O1 = []
for i in range(30):
    comb_O1.append([16.0*(i+1),0.07])
    
for j in range(500): #116
    comb_O1.append([1.+.5*(j+20),0.02])    

comb_O1 = np.array(comb_O1)
notches_O1 = np.append(notches_O1,comb_O1,axis = 0)

no_O1, sig_O1 = notches_O1.T[0],  notches_O1.T[1]
 