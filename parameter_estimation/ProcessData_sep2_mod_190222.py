#!/usr/bin/env python
# coding: utf-8


import warnings  
warnings.filterwarnings("ignore")  


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#######################################################################################################################
#                                        THIS IS THE PLACE YOU CAN CHANGE INPUTS                                      #

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
	"190222meso_mCherry_mesophyll.csv",
	"190222meso_mCherry_mesophyll.csv",
	"190222meso_mCherry_mesophyll.csv",
	"190222meso_mCherry_mesophyll.csv"
]

BSTEPS = 1000
NBINS = 20
Lin_Log_Equi = None #True for equiprobable binning, False for lineary space bins, and None for logspace binning.

#modes = ["NOT_GATED","mCherry_GATED", "EGFP_GATED", "Double_GATED","Double_OR_GATED"]
modes = ["mCherry_GATED"]

#Columns in file
column_xaxis = 'YL2-H :: mCherry-H'
column_yaxis = 'BL1-H :: GFP(488)-H'
column_xaxis_alternative_name = 'mCherry-H'
column_yaxis_alternative_name = 'GFP(488)-H'
column_phi_axis = 'FSC-H'

MIN_POINTS = 200
                                                                                                                      #
#######################################################################################################################
 
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def removeOutliers(x,lower=1,upper=99):
    a = np.array(x)
    if len(a)>0:
        p1 = a[:,0]
        upper_quartile = np.percentile(p1, upper)
        lower_quartile = np.percentile(p1, lower)
        quartileSet = (lower_quartile, upper_quartile)
        resultList = []
        for y in a.tolist():
            if y[0] >= quartileSet[0] and y[0] <= quartileSet[1]:
                resultList.append(y)
        return resultList
    else:
        return []
        
def Ix_x(p1, p2):
    p1m = np.mean(p1)
    p2m = np.mean(p2)
    
    try:
        p2 = np.std(p1)*(p2 - p2m)/np.std(p2) + p1m
    except:
        p2 = p2*p1m/p2m
    p2m = np.mean(p2)
            
    CV2tot = np.var(p1)/(p1m**2)
    CV2ext = np.cov(p1,p2)[0,1]/(p1m*p2m)
    CV2int = CV2tot - CV2ext
    return CV2int, p1m, CV2tot, CV2ext

def bootsrapped_step(data2,negs,mbins,points,rr,equiprob=False):

    try:
        dlen = len(data2[column_xaxis])
        dlen2 = len(negs[column_xaxis])
    except:
        dlen = len(data2[column_xaxis_alternative_name])
        dlen2 = len(negs[column_xaxis_alternative_name])
    
    try:    
        EGFPmChe = data2[column_xaxis]# + data2[column_yaxis]
        EGFPmChe2 = negs[column_xaxis]# + negs[column_yaxis]
    except:
        EGFPmChe = data2[column_xaxis_alternative_name]# + data2[column_yaxis_alternative_name]
        EGFPmChe2 = negs[column_xaxis_alternative_name]# + negs[column_yaxis_alternative_name]
            
    np.random.seed(int(rr*100))
    sampling = sorted(np.random.choice(list(range(dlen)), int(dlen/1)))
    sampling2 = sorted(np.random.choice(list(range(dlen2)), int(dlen2/1)))

    if mbins == None:
        mbins = int(np.sqrt(dlen))
    data = {}
    Ndata = {}
    
    try:
        data[column_xaxis] = np.array(data2[column_xaxis])[sampling]
        data[column_yaxis] = np.array(data2[column_yaxis])[sampling]
        Ndata[column_xaxis] = np.array(negs[column_xaxis])[sampling2]
        Ndata[column_yaxis] = np.array(negs[column_yaxis])[sampling2]
    except:
        data[column_xaxis] = np.array(data2[column_xaxis_alternative_name])[sampling]
        data[column_yaxis] = np.array(data2[column_yaxis_alternative_name])[sampling]
        Ndata[column_xaxis] = np.array(negs[column_xaxis_alternative_name])[sampling2]
        Ndata[column_yaxis] = np.array(negs[column_yaxis_alternative_name])[sampling2]
        
    data[column_phi_axis] = np.array(data2[column_phi_axis])[sampling]
    Ndata[column_phi_axis] = np.array(negs[column_phi_axis])[sampling2]

    try:
        EGFPmChe = np.array(EGFPmChe)[sampling]
        EGFPmChe2 = np.array(EGFPmChe2)[sampling2]
    except Exception as e:
        print(e)
        
    if equiprob == False:
        xmin = min(EGFPmChe)
        xmax = max(EGFPmChe)
        bins = list(np.linspace(xmin, xmax, mbins+1))        
    elif equiprob == True:   
        xlen = len(EGFPmChe) 
        xss = sorted(EGFPmChe)
        xnum = int(xlen/mbins)
        bins = []
        ih = 0
        while ih<mbins+1:
            try:
                bins.append(xss[ih*xnum])
            except:
                pass
            ih=ih+1

        while len(bins)<mbins+1:
            bins.append(bins[-1]+0.01)
            
    else:
        xmin = 4.39397056e+02
        xmax = 25000
        #xmin = min(EGFPmChe) 
        #xmax = max(EGFPmChe) 
        #bins = list(np.exp(np.linspace(np.log(xmin),np.log(xmax),mbins+2)))
        bins = np.logspace(np.log10(xmin),np.log10(xmax),num=mbins+1,base=10)
            
    digitized = np.digitize(EGFPmChe, bins)
    negs_digz = np.digitize(EGFPmChe2, bins)

    Ubins = {}
    for ih in range(len(digitized)):
        item = digitized[ih]
        if item in Ubins:
            try:
                Ubins[item].append([data[column_yaxis][ih],data[column_xaxis][ih],data[column_phi_axis][ih]])
            except:
                pass
        else:
            try:
                Ubins[item] = [[data[column_yaxis][ih],data[column_xaxis][ih],data[column_phi_axis][ih]]]
            except:
                pass
                
    Ubins2 = {}
    for ih in range(len(negs_digz)):
        item2 = negs_digz[ih]
        if item2 in Ubins2:
            try:
                Ubins2[item2].append([Ndata[column_yaxis][ih],Ndata[column_xaxis][ih],Ndata[column_phi_axis][ih]])
            except:
                pass
        else:
            try:
                Ubins2[item2] = [[Ndata[column_yaxis][ih],Ndata[column_xaxis][ih],Ndata[column_phi_axis][ih]]]
            except:
                pass
                
    #for item in Ubins:
    #    dlen2 = len(Ubins[item])
    #    sampling2 = sorted(np.random.choice(list(range(dlen2)), int(dlen2/2)))
    #    Ubins[item] = np.array(Ubins[item])[sampling2]   
  
    Gn = [np.nan]*mbins
    Mc = (np.array(bins[0:-1])+np.array(bins[1:]))/2.0
    Vr = [np.nan]*mbins
    Cv = [np.nan]*mbins
    Sd = [np.nan]*mbins
    Rm = [np.nan]*mbins
    Pm = [np.nan]*mbins
    Cvt = [np.nan]*mbins
    Cve = [np.nan]*mbins
    Cvs = [np.nan]*mbins
    
    for item in Ubins:
        if item==0:
            pass
        elif item in Ubins2:
            x = np.array(removeOutliers(np.array(Ubins2[item])))
            #x = np.array(Ubins2[item])
            if len(x)>points:
                e1, e2 = x[:,0], x[:,1]
                phi_e = x[:,2]
                            
                ind = item-1
                try:
                    x = np.array(removeOutliers(np.array(Ubins[item])))
                    #x = np.array(Ubins[item])
                    if len(x)>points:
                        try:
                            
                            t1 = x[:,0]
                            t2 = x[:,1]
                            
                            phi_t = x[:,2]                                
                            It_t, tm, cv2t_tot, cv2t_ext = Ix_x(t1, t2)
                          
                            Ie_e, em, cv2e_tot, cv2e_ext = Ix_x(e1, e2)
                            dele_e = np.mean(e1) - np.mean(e2)
                            sm = tm - em
 
                            Nw2 = ( np.cov(t1,phi_t)[0,1] - np.cov(e1,phi_e)[0,1] \
                            )/(sm*np.mean([np.mean(phi_t),np.mean(phi_e)]))        
                            Is_s = It_t*(tm/sm)**2 - Ie_e*(em/sm)**2 #- (1+2*em/sm)*0.02 #- Nw2*dele_e/sm
                            

                            CV2_stot = cv2t_tot*(tm/sm)**2 - cv2e_tot*(em/sm)**2 - 2*(em/sm)*Nw2
                            
                            if CV2_stot>0 and sm>0:                               
                                Gn[ind] = sm
                                Vr[ind] = CV2_stot*sm**2
                                Cv[ind] = 100*np.sqrt(CV2_stot)
                                Sd[ind] = np.std(t1)
                                Rm[ind] = 100*(len(Ubins[item])-len(x))/len(Ubins[item])
                                Pm[ind] = len(x)  
                                Cve[ind] = cv2e_tot
                                Cvt[ind] = cv2t_tot 
                                Cvs[ind] = CV2_stot
                                                                   
                        except:
                            pass    
                except:
                    pass    
    #print("KKKKK")
    return [Gn, Cv, Vr, Mc, Sd, Rm, Pm, Cvt, Cve, Cvs]

def process_file(data2,negs,mbins=100,points=1,bsteps = 50,equiprob=False):
    pool = multiprocessing.Pool(min(bsteps,round(0.9*multiprocessing.cpu_count())))
                    
    gGn = []
    gMc = []
    gVr = []
    gCv = []
    gSd = []
    gRm = []
    gPm = []
    gCvt = []
    gCve = []
    gCvs = []
    ulen = 1.0e+20      
    
    rands = [ x*np.random.rand() for x in range(bsteps) ]
    results = [
        pool.apply_async( 
            bootsrapped_step, 
            args = (data2, negs, mbins, points, rands[ig],equiprob)
        ) for ig in range(bsteps)
    ]
    
    rows = [ result.get() for result in results ]
    for row in rows:
        try:
            Gn, Cv, Vr, Mc, Sd, Rm, Pm, Cvt, Cve, Cvs = row
            gGn.append(Gn)
            gMc.append(Mc)
            gVr.append(Vr)
            gCv.append(Cv)
            gSd.append(Sd)   
            gRm.append(Rm)
            gPm.append(Pm)    
            gCvt.append(Cvt)
            gCve.append(Cve)
            gCvs.append(Cvs)
        except:
            pass    

    gGn = np.array(gGn)
    gMc = np.array(gMc)
    gVr = np.array(gVr)
    gCv = np.array(gCv)
    gSd = np.array(gSd)
    gRm = np.array(gRm)
    gPm = np.array(gPm)
    gCvt = np.array(gCvt)
    gCve = np.array(gCve)
    gCvs = np.array(gCvs)
    
    pool.close()
    
    return (
        np.nanmean(gGn,0),
        np.nanmean(gCv,0),
        np.nanstd(gCv,0),
        np.nanmean(gSd,0),
        np.nanmean(gMc,0),
        np.nanmean(gVr,0),
        np.nanstd(gVr,0),
        np.nanmean(gRm,0),
        np.nanmean(gPm,0),
        np.nanmean(gCvt,0),
        np.nanmean(gCve,0),
        np.nanmean(gCvs,0)
    )

if __name__ == '__main__': ########################################################################################

    for mode in modes:
        mode_name = mode + "_zero_point_correction_mod"
        data = []
        ind = 0
        names = files
        negatives = {}
        
        for file in files:
            print(file)
            
            negatives = {}
            
            try:
                data3 = pd.read_csv(file)
            except:
                data3 = pd.read_excel(file)
                
            try:
                if mode == "NOT_GATED":
                    data2 = data3[ (data3[column_xaxis]>0) & (data3[column_yaxis]>0) ]                      
                elif mode == "mCherry_GATED":
                    data2 = data3[ (data3[column_xaxis]>thr[ind][0]) & (data3[column_yaxis]>0) ]       
                elif mode == "EGFP_GATED":
                    data2 = data3[ (data3[column_xaxis]>0) & (data3[column_yaxis]>thr[ind][1]) ]  
                elif mode == "Double_GATED":
                    data2 = data3[ (data3[column_xaxis]>thr[ind][0]) & (data3[column_yaxis]>thr[ind][1]) ] 
                elif mode == "Double_OR_GATED":
                    data2 = data3[ (data3[column_xaxis]>thr[ind][0]) | (data3[column_yaxis]>thr[ind][1]) ] 
            except:
                if mode == "NOT_GATED":
                    data2 = data3[ (data3[column_xaxis_alternative_name]>0) & (data3[column_yaxis_alternative_name]>0) ]    
                elif mode == "mCherry_GATED":
                    data2 = data3[ (data3[column_xaxis_alternative_name]>thr[ind][0]) & (data3[column_yaxis_alternative_name]>0) ]            
                elif mode == "EGFP_GATED":
                    data2 = data3[ (data3[column_xaxis_alternative_name]>0) & (data3[column_yaxis_alternative_name]>thr[ind][1]) ]        
                elif mode == "Double_GATED":
                    data2 = data3[ (data3[column_xaxis_alternative_name]>thr[ind][0]) & (data3[column_yaxis_alternative_name]>thr[ind][1]) ]
                elif mode == "Double_OR_GATED":                    
                    data2 = data3[ (data3[column_xaxis_alternative_name]>thr[ind][0]) | (data3[column_yaxis_alternative_name]>thr[ind][1]) ]
            
            if negs[ind] not in negatives:
                dd = pd.read_csv(negs[ind])
                try:
                    dd = dd[(dd[column_xaxis]>0) & (dd[column_yaxis]>0)]
                except:
                    dd = dd[(dd[column_xaxis_alternative_name]>0) & (dd[column_yaxis_alternative_name]>0)]
                    
                negatives[negs[ind]] = dd
            else:
                pass
                    
            try:
                dd = process_file(data2,negatives[negs[ind]],mbins=NBINS,points=MIN_POINTS,bsteps=BSTEPS,equiprob=Lin_Log_Equi)
                data.append(dd)
            except:
                pass
            ind = ind + 1
                
        index = 0
        r1file = open(mode_name+".txt","w")
        labels = "EGFP mean\tEGFP mean %Cv\tsd EGFP mean %Cv\tsd EGFP mean\tEGFP+mCherry groupd\tEGFP \
        variance\tsd EGFP variance\t% discarded\t# cells\tCV^2-t_tot\tCV^2-e_tot\tCV^2-s_tot\n"
        
        #bins = np.logspace(np.log10(10**0),np.log10(10**7),num=NBINS+2,base=10)
        #Mc = (np.array(bins[0:-1])+np.array(bins[1:]))/2.0
        #Mc = Mc[0:-1]        
        
        for row in data:
            r1file.write(files[index]+"\n")
            r1file.write(labels)
            Gn, Cv, eCv, Sd, Mc, Vr, eVr, Rm, Pm, Cvt, Cve, Cvs = row
            dlen = len(Gn)
            for i in range(dlen):
                cols = [Gn[i],Cv[i],eCv[i],Sd[i],Mc[i],Vr[i],eVr[i],Rm[i],Pm[i], Cvt[i], Cve[i], Cvs[i]]
                cols = "\t".join([str(ss) for ss in cols])
                r1file.write(cols+"\n")
            index = index + 1
            r1file.write("\n")
        r1file.close()       
        
        num_plot_mix = 2        
        plt.figure(figsize=(7,10))
        names = files
        ind = 0
        ind2 = 0
        mColor = ['#01adf7','#ffa333','#fe01fa'] #['#01adf7','#fe01fa',,'#ffa333','#875d40']
        for row in data:
            Gn, Cv, eCv, Sd, Mc, Vr, eVr, Rm, Pm, Cvt, Cve, Cvs = row
            Gn, Cv, eCv = zip(*sorted(zip(Gn, Cv, eCv)))
            Gn = np.array(Gn)
            Cv = np.array(Cv)
            eCv = np.array(eCv)
            if (ind+1)%num_plot_mix==1:
                plt.subplot(3, 1, ind2+1)
                #[ plt.axvline(b,ls="--",lw=0.5,color='red') for b in Gn ]
            plt.plot(Gn,Cv,color=mColor[ind%num_plot_mix],linestyle=':',linewidth=2)
            plt.errorbar(Gn, Cv, eCv, color=mColor[ind%num_plot_mix], label=names[ind], linestyle='None',linewidth=1,capsize=1,marker='o',ms=5) 
            if (ind+1)%num_plot_mix==0:
                plt.tight_layout()
                plt.xscale('log')
                plt.legend()
                plt.xlabel("EGFP mean")
                plt.ylabel("%CV EGFP")
                #plt.xlim(400,200000)
                #plt.ylim(0,min(150,np.nanmax(Cv)*1.1))   
            ind = ind+1
            if ind%num_plot_mix==0:
                ind2 = ind2 + 1 
        plt.savefig(mode_name+".png", dpi=100)
        
        plt.figure(figsize=(7,10))
        names = files
        ind = 0
        ind2 = 0
        mColor = ['#01adf7','#ffa333','#fe01fa'] #['#01adf7','#fe01fa',,'#ffa333','#875d40']
        for row in data:
            Gn, Cv, eCv, Sd, Mc, Vr, eVr, Rm, Pm, Cvt, Cve, Cvs = row
            Gn, Cv, eCv = zip(*sorted(zip(Gn, Cv, eCv)))
            Gn = np.array(Gn)
            Cv = np.array(Cv)
            eCv = np.array(eCv)
            if (ind+1)%num_plot_mix==1:
                plt.subplot(3, 1, ind2+1)
                #[ plt.axvline(b,ls="--",lw=0.5,color='red') for b in Gn ]
            plt.plot(Gn,Cvt,label=names[ind]+"CV2t")
            plt.plot(Gn,Cve,label=names[ind]+"CV2e")
            plt.plot(Gn,Cvs,label=names[ind]+"CV2s")
            if (ind+1)%num_plot_mix==0:
                plt.tight_layout()
                plt.xscale('log')
                plt.legend()
                plt.xlabel("EGFP mean")
                plt.ylabel("CV^2 contribution")
                #plt.xlim(400,200000)
                #plt.ylim(0,min(5,np.nanmax(Cvt)*1.1))   
            ind = ind+1
            if ind%num_plot_mix==0:
                ind2 = ind2 + 1 
        plt.savefig(mode_name+"_CV2.png", dpi=100)
        
        """
        plt.figure(figsize=(7,10))
        names = files
        ind = 0
        ind2 = 0
        mColor = ['#01adf7','#ffa333','#fe01fa'] #['#01adf7','#fe01fa',,'#ffa333','#875d40']
        for row in data:
            Gn, Cv, eCv, Sd, Mc, Vr, eVr, Rm, Pm, Cvt, Cve, Cvs = row
            Mc, Cv, eCv = zip(*sorted(zip(Mc, Cv, eCv)))
            Mc = np.array(Mc)
            Cv = np.array(Cv)
            eCv = np.array(eCv)
            if (ind+1)%num_plot_mix==1:
                plt.subplot(3, 1, ind2+1)
                [ plt.axvline(b,ls="--",lw=0.5,color='red') for b in Mc ]
            plt.plot(Mc,Gn,color=mColor[ind%num_plot_mix],linestyle=':',linewidth=2)
            plt.errorbar(Mc, Gn, Sd, color=mColor[ind%num_plot_mix], label=names[ind], linestyle='None',linewidth=1,capsize=1,marker='o',ms=5) 
            if (ind+1)%num_plot_mix==0:
                plt.tight_layout()
                plt.xscale('log')
                plt.yscale('log')
                plt.legend()
                plt.xlabel("EGFP + mCherry")
                plt.ylabel("EGFP mean")
                #plt.xlim(400,200000)
                #plt.ylim(0,50)   
            ind = ind+1
            if ind%num_plot_mix==0:
                ind2 = ind2 + 1 
        plt.savefig(mode_name+"_sup.png", dpi=100)
        """
