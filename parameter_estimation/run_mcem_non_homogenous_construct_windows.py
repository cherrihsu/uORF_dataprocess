#!/usr/bin/env python
# coding: utf-8

# # Metropolis Hasting Algorithm + Expectation Maximization Algorithm

# # Personally coded by Erickson Fajiculay


import warnings
warnings.filterwarnings('ignore')


import numpy as np
import random as rn
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import linalg
import math
import time
import multiprocessing as mp


def log_likelihood(ks,custom_function,args=None):
    return custom_function(ks,args)

def prior(x,std=0):
    return 1.0

def SSE_conc(ks,custom_function,args=None): 
    return log_likelihood(ks,custom_function,args)

def MH_call(
    sd=np.random.uniform(0,1),
    maxiter=50,
    inner_loop=1000,
    lenks=3,
    positive_only=False,
    likelihood=None,
    args=None,
    vals=None,
    thr=1.0e-10,
	lst=None
):

    np.random.seed(int(sd*100))
    sd = np.random.uniform(0,1)  

    stds = [0]*lenks   
    for i in range(lenks):
        if positive_only:        
            stds[i] = abs(np.random.normal(sd,0.01))
        else:  
            stds[i] = abs(np.random.normal(sd,100))    

    kso = [0]*lenks 
    for i in range(lenks):
        if positive_only:
            kso[i] = np.random.lognormal(0,stds[i])
        else:
            kso[i] = np.random.normal(0,stds[i])
    if vals != None:
        kso = vals[0]
        stds= vals[1]       

    valo = [0]*lenks 
    valp = [0]*lenks 
    for i in range(lenks):
        valo[i] = 1
        valp[i] = 2

    Ks = []
    for i in range(lenks):
        Ks.append([])

    Rs  = []

    vmax = 1.0e+100
    iteration = 0
    test = 1000
    error = 1000
    sigma = 1

    delta = int(np.ceil(inner_loop/(maxiter-0.33*maxiter)))
    inner_loop_orig = inner_loop
    inner_loop = delta
    while iteration<maxiter:
        if lst:
            print("stopped")            
            return (1e+100, 1e+100, 1e+100)
        kis = []
        for i in range(lenks):
            kis.append([])	

        R = vmax*log_likelihood(kso,likelihood,args)		

        for i in range(inner_loop):
            if lst:
                break  
            ksp = [0]*lenks
            for j in range(lenks):
                if positive_only:
                    ksp[j] = np.random.lognormal(np.log(kso[j]),stds[j])
                else:
                    ksp[j] = np.random.normal(kso[j],stds[j]) 

            L = vmax*log_likelihood(ksp,likelihood,args)			 

            U = np.random.uniform(0,1)
            check = np.exp(L-R)
            if check>U:

                for j in range(lenks):
                    kis[j].append(ksp[j])
                    kso[j] = ksp[j]

                Rs.append(L)
                R = L
            else:
                for j in range(lenks):
                    kis[j].append(kso[j])	  


        for i in range(lenks):
            if len(kis[i])>1:
                [ Ks[i].append(k) for k in kis[i] ]
                if positive_only:
                    stds[i] = np.std(np.log(kis[i]))
                else:
                    stds[i] = np.std(kis[i])
                valo[i] = valp[i]
                valp[i] = np.array(kis[i]).mean()
            else:
                stds[i] = abs(np.random.normal(2*stds[i],0.1*stds[i]))

        test = 0
        for i in range(lenks):
            test = test + abs((valo[i]-valp[i])/valo[i])

        if test < 1.0e-15:
            break

        iteration = iteration + 1
        inner_loop = min(inner_loop_orig,inner_loop + delta)

    error = max(error,SSE_conc(valp,likelihood,args))
    if error<thr:
        #print(maxiter,inner_loop)
        lst.append(1)
    else:
        pass#print("processing",maxiter,inner_loop,error)
    return (valp, error, stds)

def run_MCEM(chains,params,maxiter=5,inner_loop=5*1000,positive_only=False,likelihood=None,args=None,vals=None,thr=None,lst=None):
    pool = mp.Pool(chains)
    results = [pool.apply_async( MH_call, args = (np.random.uniform(0,1),maxiter+(ih+1),inner_loop+(ih+1),params,positive_only,likelihood,args,vals,thr,lst) ) for ih in range(chains)]

    ff = [ result.get() for result in results ]
    pool.close()

    ff = [ result.get() for result in results ]
    er = []
    for x in ff:
        er.append(x[1])

    er_min = max(er)
    ks = []
    sd = []    
    for x in ff:
        if abs(x[1])<=er_min:
            ks = x[0]
            sd = x[2]

    return (ks,er_min)

def calc_construct(ks):
    k_m, r_m, k_p1, r_p1, r_p2, k_on, k_off, r_mu, k_p2 = ks
    
    A           = k_off*r_m + k_on*r_mu + k_p2*r_m + r_m*r_mu
    B           = (r_m+r_p2)*(k_off+k_p2)+(r_mu+r_p2)*(k_on+r_m+r_p2)

    P_uORF      =  k_m*k_on*k_p2/(r_p2*A)
    FF_uORF     =  1 + k_on*k_p2/B

    P_uORFm     =  k_m*k_p1/(r_m*r_p1)
    FF_uORFm    =  1 + k_p1/(r_m+r_p1)
    
    mf = k_m*(k_off + k_p2 + r_mu)/A
    mu = k_m*k_on/A
    m = k_m/r_m
    
    return [P_uORF, P_uORFm, FF_uORF, FF_uORFm, mf, mu, m]

def custom_likelihood(ks,data):
    k_m, r_m, k_p1, r_p1, b, k_on, k_off, a, k_p2 = ks

    r_m   = np.log(2)/2.5  
    r_mu  = a*r_m
    r_p1  = np.log(2)/27
    r_p2  = b*r_p1   

    k_p1 = (data["E"][3]-1)*(r_m+r_p1)    
    k_m = data["E"][1]*(r_m*r_p1)/k_p1
         
    obj = 1000*a**2 + (1/b)**2  # force a << 1 and b >> 1
    
    #w2z and s2z
    ks = [k_m, r_m, k_p1, r_p2, r_p2, k_on, k_off, r_mu, k_p2]
    P_uORF, P_uORFm, FF_uORF, FF_uORFm, mf, mu, m = calc_construct(ks)
    Pobj_uORF   = data["Z"][0] - P_uORF
    Pobj_uORFm  = data["Z"][1] - P_uORFm
    
    FFobj_uORF  = data["Z"][2] - FF_uORF
    FFobj_uORFm = data["Z"][3] - FF_uORFm
    obj = obj + np.sum(Pobj_uORF**2)+np.sum(Pobj_uORFm**2)+np.sum(FFobj_uORF**2)+np.sum(FFobj_uORFm**2) + 1000*(mu - m - mf)**2
    
    #w2e and s2e
    ks = [k_m, r_m, k_p1, r_p1, r_p1, k_on, k_off, r_mu, k_p2]
    P_uORF, P_uORFm, FF_uORF, FF_uORFm, mf, mu, m = calc_construct(ks)
    Pobj_uORF   = data["E"][0] - P_uORF
    Pobj_uORFm  = data["E"][1] - P_uORFm
    
    FFobj_uORF  = data["E"][2] - FF_uORF
    FFobj_uORFm = data["E"][3] - FF_uORFm
    obj = obj + np.sum(Pobj_uORF**2)+np.sum(Pobj_uORFm**2)+np.sum(FFobj_uORF**2)+np.sum(FFobj_uORFm**2) + 1000*(mu - m - mf)**2

    return -obj
	
if __name__ == '__main__':
	t_o = time.time()
	manager = mp.Manager()
	lst = manager.list()

	true_vals = {
		"213" : {
			"E" : [1042365.255, 12088471.65, 557416.8747, 4629552.484],
			"Z" : [427976.9176, 4095611.342, 427264.8637, 2880339.082]
		},
		"308" : {
			"E" : [1215180.928, 15620691.74, 571665.7182, 6396252.458],
			"Z" : [535146.4622, 6012037.102, 375028.472 , 4449153.346]
		},
		"222" : {
			"E" : [1441183.619, 21235397.24, 879807.4842, 11735063.89],
			"Z" : [553433.8808, 7234593.941, 664590.7569, 8372865.018]
		}
	}

	gfile = open("param_out_fixed_separate_replicates_old_flow2.txt","w")
	for name in ["213","308","222"]:
		gfile.write("\n"+name+"\n")
		f = 5
		k = 10
		thr = 1.0e-10
		ks, sd = run_MCEM(f,9,f,k*f,positive_only=True,likelihood=custom_likelihood,args=true_vals[name],thr=thr,lst=lst)
		er = abs(custom_likelihood(ks,true_vals[name]))

		inner_loop = 0

		while er>thr:
			ks, sd = run_MCEM(f,9,f,k*f,positive_only=True,likelihood=custom_likelihood,args=true_vals[name],thr=thr,lst=lst)
			er = abs(custom_likelihood(ks,true_vals[name]))
			#print(ks,er,f,k*f)

			gfile.write("\t".join([ str(x) for x in ks +[ " additional info = ", er, f, k*f ]])+"\n")
			inner_loop = inner_loop + k*f
			f = f+1
			if f-1 == 10:
				k = k*10
				f = 5

			print(inner_loop)
			if inner_loop>110000:
				break

		print(ks)
		print("error =",er)
		print(time.time()-t_o)
	gfile.close()