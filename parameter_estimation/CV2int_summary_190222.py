#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

##############################################################################################################

files = [ 
    "meso_1uM16.W2Z_mesophyll.csv",
    "meso_1uM16.W2E_mesophyll.csv",
    "meso_1uM16.S2Z_mesophyll.csv",
    "meso_1uM16.S2E_mesophyll.csv"
]

thr = [
    [457, 823],
    [457, 823],
    [457, 823],
    [457, 823]
]

negs = [
    "190222meso_mCherry_mesophyll.csv",
    "190222meso_mCherry_mesophyll.csv",
    "190222meso_mCherry_mesophyll.csv",
    "190222meso_mCherry_mesophyll.csv"
]

NUM_BSTEPS = 100
mol_counts = True

column_xaxis = 'mCherry-H'
column_yaxis = 'GFP(488)-H'
column_phi_axis = 'FSC-H'

##############################################################################################################

data = []
negatives = []
    
ind = 0
for file in files:
    print("Reading file "+ file +" : It may take a while")
    data.append(pd.read_csv(file))
    negatives.append(pd.read_csv(negs[ind]))
    ind = ind + 1
print()

names = files

R_summary = open("CVsummary.txt","w")
R_summary.write("\
file_name\t\
<t>\t\
<s>\t\
<theta>\t\
CV^2-cell_size\t\
It_t\t\
Ie_e\t\
Is_s\t\
CV^2-t_tot\t\
CV^2-t_ext\t\
CV^2-e_tot\t\
CV^2-e_ext\t\
CV^2-s_tot\t\
CV^2-s_ext\
\n") 

def removeOutliers(x,lower=1,upper=99):
    if len(x)>0:
        p1 = x[column_yaxis]
        upper_quartile = np.percentile(p1, upper)
        lower_quartile = np.percentile(p1, lower)
        quartileSet = (lower_quartile, upper_quartile)
        return x[ (x[column_yaxis] >= quartileSet[0]) & (x[column_yaxis]<= quartileSet[1]) ]

def Ix_x(p1, p2):
    p1m = np.mean(p1)
    p2m = np.mean(p2)
    
    p2 = np.std(p1)*(p2 - p2m)/np.std(p2) + p1m
    p2m = np.mean(p2)
            
    CV2tot = np.var(p1)/(p1m**2)
    CV2ext = np.cov(p1,p2)[0,1]/(p1m*p2m)
    CV2int = CV2tot - CV2ext
    return CV2int, p1m, CV2tot, CV2ext

def MESF_counts(x):
    return 1501.7611318956613*x -340544.8455004525
    
ind = 0
for row3 in data:
    print("processing "+names[ind])
    rowT = row3[(row3[column_xaxis]>=thr[ind][0]) & (row3[column_yaxis]>=0)]
    #rowt = removeOutliers(rowT)
    rowN = negatives[ind][(negatives[ind][column_xaxis]>=0) & (negatives[ind][column_yaxis]>=0)]
    
    dlenT = len(rowT)
    dlenN = len(rowN)
    
    if len(rowT)>0:
    
        computed_row = []
        for step in range(NUM_BSTEPS):
            samplingT = sorted(np.random.choice(list(range(dlenT)), int(dlenT)))
            samplingN = sorted(np.random.choice(list(range(dlenN)), int(dlenN)))

            phi_t = np.array(rowT[column_phi_axis])[samplingT]
            t1 = np.array(rowT[column_yaxis])[samplingT]
            t2 = np.array(rowT[column_xaxis])[samplingT]    

            if mol_counts == True:
                phi_t = MESF_counts(phi_t)
                t1 = MESF_counts(t1)
                t2 = MESF_counts(t2)

            phi_e = np.array(rowN[column_phi_axis])[samplingN]
            e1 = np.array(rowN[column_yaxis])[samplingN]
            e2 = np.array(rowN[column_xaxis])[samplingN]
            
            if mol_counts == True:
                phi_e = MESF_counts(phi_e)
                e1 = MESF_counts(e1)
                e2 = MESF_counts(e2)

            It_t, tm, CV2t_tot, CV2t_ext = Ix_x(t1, t2)
            Ie_e, em, CV2e_tot, CV2e_ext = Ix_x(e1, e2)
            dele_e = np.mean(e1) - np.mean(e2)
            sm = tm - em

            Nw2 = ( np.cov(t1,phi_t)[0,1] - np.cov(e1,phi_e)[0,1] )/(sm*np.mean([np.mean(phi_t),np.mean(phi_e)]))        
            Is_s = It_t*(tm/sm)**2 - Ie_e*(em/sm)**2 - (1+2*em/sm)*0.02 #- Nw2*dele_e/sm

            CV2s_int = Is_s
            CV2s_tot = CV2t_tot*(tm/sm)**2 - CV2e_tot*(em/sm)**2 - 2*(em/sm)*Nw2
            CV2s_ext = CV2s_tot - CV2s_int 
   
            computed_row.append([tm, sm, em, Nw2, It_t, Ie_e, Is_s,  CV2t_tot, CV2t_ext, CV2e_tot, CV2e_ext, CV2s_tot, CV2s_ext])
        
        computed_row = np.array(computed_row)
        
        row_mean = np.mean(computed_row,axis=0)
        vals = [names[ind]+"_mean", *row_mean ]
        R_summary.write( "\t".join([str(x) for x in vals])+"\n" )
        
        row_stds = np.std(computed_row,axis=0)
        vals = [names[ind]+"_stdv", *row_stds ]
        R_summary.write( "\t".join([str(x) for x in vals])+"\n" )
        
    ind = ind+1
R_summary.close()