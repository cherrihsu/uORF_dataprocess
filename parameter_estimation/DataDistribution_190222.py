#!/usr/bin/env python
# coding: utf-8
#
# Plotting data distribution

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.optimize import curve_fit

##################################################################################################################
#                                    PART OF THE CODE WHERE WE CAN CHANGE THE INPUTS                                 #
    
files = [ 
    "meso_1uM16.W2Z_mesophyll.csv",
    "meso_1uM16.S2Z_mesophyll.csv",
    "meso_1uM16.W2E_mesophyll.csv",
    "meso_1uM16.S2E_mesophyll.csv"
]

thr = [
    [457, 823],
    [457, 823],
    [457, 823],
    [457, 823]
]

negs = [
    "190222meso_PEG_mesophyll.csv",
    "190222meso_PEG_mesophyll.csv",
    "190222meso_PEG_mesophyll.csv",
    "190222meso_PEG_mesophyll.csv"
]

mode0 = "NOT_gated"
mode1 = "double_gated"
mode2 = "mcherry_gated"
mode3 = "eGFP_gated"
mode4 = "double_OR_gated"

Machine_Filtered = False
mCherry_dist = True

modes = [mode2]

# For machine filtering (Machine learning)
DBScan_eps=25
DBScan_min_samples=50

#Columns in file 
column_xaxis = 'YL2-H :: mCherry-H'
column_yaxis = 'BL1-H :: GFP(488)-H'
column_xaxis_alternative_name = 'mCherry-H'
column_yaxis_alternative_name = 'GFP(488)-H'

##################################################################################################################


def removeOutliers(x,lower=1,upper=99):
    a = np.array(x)
    upper_quartile = np.percentile(a, upper)
    lower_quartile = np.percentile(a, lower)
    quartileSet = (lower_quartile, upper_quartile)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return resultList

def gauss(x, *p):
    A, mu, sigma = p
    return A*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*( (x-mu)/sigma )**2)

def cutoffs(x):
    a = np.array(x)
    upper_quartile = np.percentile(a, 99)
    lower_quartile = np.percentile(a, 1)
    return [lower_quartile,upper_quartile]
    
for mode in modes: 
    mode_name = mode + ("_Machine_Filtered" if Machine_Filtered else "") + ("_mCherry_dist" if mCherry_dist else "_EGFP_dist")
    data = []
    ind = 0
    names = files
    negatives = {}
    collected = []

    R0 = open(mode_name+".txt","w")
    R0.write("construct\tP-mean\tP-stdv\tP-var\tFfactor\t%CV\tlen(data)\n")
    
    print()
    
    for file in files:
        print(file)
        
        negatives = {}
        
        try:
            data3 = pd.read_csv(file)
        except:
            data3 = pd.read_excel(file)
            
        try:
            if mode == mode0:
                data2 = data3[ (data3[column_xaxis]>0) & (data3[column_yaxis]>0) ]    
            elif mode == mode1:
                data2 = data3[ (data3[column_xaxis]>thr[ind][0]) & (data3[column_yaxis]>thr[ind][1]) ]                 
            elif mode == mode2:
                data2 = data3[ (data3[column_xaxis]>thr[ind][0]) & (data3[column_yaxis]>0) ]       
            elif mode == mode3:
                data2 = data3[ (data3[column_xaxis]>0) & (data3[column_yaxis]>thr[ind][1]) ]  
            elif mode == mode4:
                data2 = data3[ (data3[column_xaxis]>thr[ind][0]) | (data3[column_yaxis]>thr[ind][1]) ] 
        except:
            if mode == mode0:
                data2 = data3[ (data3[column_xaxis_alternative_name]>0) & (data3[column_yaxis_alternative_name]>0) ]
            elif mode == mode1:
                data2 = data3[ (data3[column_xaxis_alternative_name]>thr[ind][0]) & (data3[column_yaxis_alternative_name]>thr[ind][1]) ]                
            elif mode == mode2:
                data2 = data3[ (data3[column_xaxis_alternative_name]>thr[ind][0]) & (data3[column_yaxis_alternative_name]>0) ]            
            elif mode == mode3:
                data2 = data3[ (data3[column_xaxis_alternative_name]>0) & (data3[column_yaxis_alternative_name]>thr[ind][1]) ]        
            elif mode == mode4:                    
                data2 = data3[ (data3[column_xaxis_alternative_name]>thr[ind][0]) | (data3[column_yaxis_alternative_name]>thr[ind][1]) ]
                
        if Machine_Filtered:
            if negs[ind] not in negatives:
                dd = pd.read_csv(negs[ind])
                try:
                    dd = dd[(dd[column_xaxis]>0) & (dd[column_yaxis]>0)]
                    x, y = dd[column_xaxis], dd[column_yaxis]
                except:
                    dd = dd[(dd[column_xaxis_alternative_name]>0) & (dd[column_yaxis_alternative_name]>0)]
                    x, y = dd[column_xaxis_alternative_name], dd[column_yaxis_alternative_name]
                    
                x, y = np.array(x), np.array(y)
                xy = np.vstack((x,y))
                cls = DBSCAN(eps=DBScan_eps, min_samples=DBScan_min_samples).fit_predict(xy.T)
                negatives[negs[ind]] = [xy.T,cls]
            else:
                pass
                
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            clf.fit(negatives[negs[ind]][0], negatives[negs[ind]][1])
                
            try:
                x, y = data2[column_xaxis], data2[column_yaxis]
            except:
                x, y = data2[column_xaxis_alternative_name], data2[column_yaxis_alternative_name]
            
            x, y = np.array(x), np.array(y)
            xy = np.vstack((x,y))
            res = clf.predict(xy.T)
            
            
            gave = 0
            gmax = 0
            gI = 0
            
            for g in np.unique(res):
                ##plt.scatter(x[res==g], y[res==g],label=names[ind]+"_"+str(g))
                #plt.hist(x[res==g],bins=int(1*np.sqrt(max(1,len(x[res==g])))),label=names[ind]+"MCHE",fill=False,histtype='step')
                #plt.hist(y[res==g],bins=int(1*np.sqrt(max(1,len(y[res==g])))),label=names[ind]+"EGFP",fill=False,histtype='step')
                ##plt.xscale('log')
                ##plt.yscale('log')
                #plt.legend()
                #plt.show()
                
                gave = np.mean(y[res==g])
                if gave>gmax:
                    gmax = gave
                    gI = g
            
            ##plt.legend()
            ##plt.show()
                
            row = y[res==gI] if not mCherry_dist else x[res==gI]
        else:
            try:
                row = data2[column_yaxis] if not mCherry_dist else data2[column_xaxis]
            except:
                row = data2[column_yaxis_alternative_name] if not mCherry_dist else data2[column_xaxis_alternative_name]
            
        #collected.append(row)
        #mbins = 100 
        #mbins = int(1*np.sqrt(max(1,len(row))))
        mbins = list(np.exp(np.linspace(np.log(1),np.log(10**5),200)))
        mbins_log = np.linspace(np.log(1),np.log(10**5),200)
        
        log_row = np.log(row)
        x_a = np.mean(log_row)
        x_s = np.std(log_row)
        
        plt.subplot(3, 2, ind+1)
        plt.hist(row,bins=mbins,label=names[ind])
        
        try:
            hist, bin_edges = np.histogram(log_row, density=False,bins=mbins_log)
            bin_centres = (bin_edges[:-1] + bin_edges[1:])/2                    
            p0 = [1.0, x_a, x_s]    
            coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)    
            x_a = np.exp(coeff[1]+0.5*coeff[2]**2)
            x_v = np.exp(2*coeff[1]+coeff[2]**2)*(np.exp(coeff[2]**2)-1)
            x_s = np.sqrt(x_v)
            print(x_a/np.mean(row),x_v/np.var(row))
            x_v = round(x_v,2)
            x_ff = round(x_s**2/x_a,2)
            x_cv = round(100*x_s/x_a,2)
            x_a = round(x_a,2)
            x_s = round(x_s,2)
            
            pdf = gauss(bin_centres, *coeff)
            #plt.plot(np.exp(bin_centres),hist,label=names[ind])
            collected.append(np.exp(bin_centres))
            collected.append(hist)
            collected.append(pdf)
            plt.plot(np.exp(bin_centres),pdf,label='fit')
        except:
            x_a = round(np.mean(row),2)
            x_s = round(np.std(row),2)
            x_v = round(np.var(row),2)
            x_ff = round(np.var(row)/np.mean(row),2)
            x_cv = round(100*np.std(row)/np.mean(row),2)
            
        plt.legend(loc="upper left", prop={'size': 5})
        plt.xscale('log')   
        plt.tight_layout()
        plt.xlim(10**2,10**5)

        
        vals = [
            names[ind], x_a, x_s, x_v, x_ff, x_cv, len(row)
        ]
        
        print("name = ",names[ind])
        print("mean = ",vals[1])
        print("stdv = ",vals[2])
        print("var  = ",vals[3])
        print("FF   = ",vals[4])
        print("% CV = ",vals[5])
        print("len  = ",vals[6])
        print()
        
        R0.write( "\t".join([str(x) for x in vals])+"\n" )
        ind = ind+1
        
    plt.savefig(mode_name+".png", dpi=100)
    plt.close()
    R0.close()


    with open(mode_name+".txt", "a") as f:
        f.write("\n")
        ind = 0
        ind2 = 0
        for row in collected:
            if ind%3==0:
                f.write("\n"+names[ind2]+"\n")
                ind2 = ind2 + 1
            f.write("\t".join([str(x) for x in row])+"\n")
            ind = ind + 1