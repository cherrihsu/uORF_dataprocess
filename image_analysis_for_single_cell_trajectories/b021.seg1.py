# -*- coding: utf-8 -*-
"""
read in seg0, for segments size > 5400 erode to different cells
"""
import os
import numpy as np
from scipy import ndimage as ndimage
from skimage import measure
from scipy.ndimage.morphology import binary_dilation as dila
from scipy.ndimage.morphology import binary_erosion as eros
import pickle

#%%
def cluster_cut(bw0):
    eros_count = 0
    seg_number = []
    bw = bw0.copy()
    seg0 = measure.label(bw)
    seg0[:,:] = 0
    
    size_limit = []
    bw_blank = bw.copy() 
    bw_blank[:,:] = False
    bw_blank[512,512] = True
    for i in range(51):
        bw_blank = dila(bw_blank)
        size_limit.append(np.sum(bw_blank))

    while True:
        bw = eros(bw)
        eros_count += 1
        seg9 = measure.label(bw)
        propsall = measure.regionprops(seg9)
        cell_sizes = []
        for i in range(len(propsall)):
            props = propsall[i]
            cell_sizes.append(props.area)
        
            
        seg_number.append(np.max(seg9))
        if np.max(seg9) == 0:
            break

        if eros_count > 51:
            max_size = 5
        else:
            max_size = size_limit[-eros_count]

        if max(cell_sizes) < max_size:
            if len(seg_number) > 1 and seg_number[-1] < seg_number[-2]:
                break
    
    eros_time = seg_number.index(np.max(seg_number))+1
    bw = bw0.copy()
    

    for i in range(eros_time):
        bw = eros(bw)
    
    seg1 = measure.label(bw)
    biggest_label = np.max(seg1)
    for i in range(1,biggest_label+1):
        bw1 = seg1 == i
        for j in range(eros_time):
            bw1 = dila(bw1)

        seg0[bw1 & bw0] = i
    return(seg0)

#%%
path = './'
seg0_file = []
for folderName, subfolders, filenames in os.walk(path):
    for filename in filenames:
        if 'pkl' in filename:
            seg0_file.append(folderName+filename)


segfile = seg0_file[0]
f = open(segfile,'rb')
seg0s = pickle.load(f)
f.close()

seg1s = []

#%% deal with frames
for frame_index in range(len(seg0s)):
    print(frame_index)
    seg0 = seg0s[frame_index].copy()
    
    propsall = measure.regionprops(seg0)
    
    for i in range(len(propsall)):
        props = propsall[i]
        bw = seg0 == props.label
        if props.area > 5400:# or ratio > 1.5:# or circularity < 0.5:
            seg_part = cluster_cut(bw)
            if np.max(seg_part) > 1:
                cut = True
                seg0[bw] = 0 
                for label in range(1,np.max(seg_part)+1):
                    bw1 = seg_part == label
                    if np.sum(bw1) >= 700:
                        seg0_label = np.max(seg0)
                        seg0[bw1] = seg0_label + 1

    seg1 = seg0.copy()
    
#%% relabelling
    seg_temp = seg1.copy()
    seg1[:,:] = 0
    seg1_label = 1
    added = []
    for row in range(seg_temp.shape[0]):
        for col in range(seg_temp.shape[1]):
            target = seg_temp[row,col]
            if target > 0 and target not in added:
                bw = seg_temp == target
                bw = ndimage.binary_fill_holes(bw).astype(bool)
                added.append(target)
                seg1[bw] = seg1_label
                seg1_label +=1    
    
    seg1s.append(seg1)

f = open(seg0_file[0][:-9]+'1_all.pkl','wb')
pickle.dump(seg1s,f)
f.close()