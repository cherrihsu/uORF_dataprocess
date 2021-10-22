# -*- coding: utf-8 -*-
"""
from one tif, many frames, find a signal threshold to differential cells and background
"""
import os
from skimage import io
import cv2
import numpy as np
import skimage
from scipy import ndimage as ndimage
from skimage import measure
from skimage.segmentation import mark_boundaries
import pickle

#%%
def label_img(img,seg):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = .8
    fontColor              = (255,255,255)
    lineType               = 2
    propsall = measure.regionprops(seg)
    img1 = np.uint8(mark_boundaries(img, seg, color=(0, 0, 1))*255)
    for i in range(len(propsall)):
        props = propsall[i]
        row = int(props.centroid[0])
        col = int(props.centroid[1])
        cv2.putText(img1,str(props.label).zfill(2), 
            (col-14,row+10), 
            font, 
            fontScale,
            fontColor,
            lineType)
    return img1

#%%
def cell_edge(img0,edge_val):

    bw = img0 > edge_val
    cell_th = skimage.filters.threshold_otsu(np.array(img0[~bw]))
    bw = img0 > cell_th
    bw = ndimage.binary_fill_holes(bw).astype(bool)
    img1 = img0.copy()
    img1[bw] = cell_th

    top = cell_th
    if top == 0:
        top = 1
    base = np.min(img1)
    img_cell = np.zeros((1024,1024,3),dtype='uint8')
    img_cell[:,:,1] = np.uint8( ( (img1[:,:]-base) / (top - base ) *255 ) )
    
    return [cell_th,img_cell]
#%% collecting tif file names

path = './data/'
ending_frame = 99
tifs = []
for folderName, subfolders, filenames in os.walk(path):
    for filename in filenames:
        if 'tif' in filename:
            tifs.append(folderName+'/'+filename)
print('tifs',len(tifs))
for tif_index in range(len(tifs)):
    im = io.imread(tifs[tif_index])
    if ending_frame > im.shape[0]:
        ending_frame = im.shape[0]
    
    pathout = tifs[tif_index][:-10]+'.seg0/'
    try:
        os.mkdir(pathout)
    except:
        print('dir exists')
    
    #%% deal with frames
    
    seg0s = []
    ths = []
    for j in range(ending_frame):
    
        img0 = im[j,:,:]
        edge_th = 255
        seg_counts = []
        edge_return = []
        stop = False
        loop_count = 0
    #%% first parameter cell_th:
        while edge_th > 3:
            
            [edge_th,img_cell] = cell_edge(img0,edge_th)
            edge_return.append(edge_th)
        
            blur_size = 3
            img_blur = cv2.medianBlur(img_cell, blur_size)
        
            blur_th = skimage.filters.threshold_otsu(np.array(img_blur[:,:,1]))
        
            bw1_cells = img_blur[:,:,1] > blur_th
        
            seg0 = measure.label(bw1_cells)
        
        
        #%% remove segments that size < 700 
            propsall = measure.regionprops(seg0)
            out_count = 0
            for i in range(len(propsall)):
                props = propsall[i]
                if props.area < 700:
                    bw = seg0 == props.label
                    seg0[bw] = 0
                    out_count += 1
            seg_counts.append(len(propsall) - out_count)

#%%
        print(seg_counts,edge_return)
        if seg_counts.index(max(seg_counts)) == 0:
            edge_starting = 255
        else:
            edge_starting = edge_return[seg_counts.index(max(seg_counts))-1]

#%%
        [edge_th,img_cell] = cell_edge(img0,edge_starting)
    
        blur_size = 3
        img_blur = cv2.medianBlur(img_cell, blur_size)
    
        blur_th = skimage.filters.threshold_otsu(np.array(img_blur[:,:,1]))
    
        bw1_cells = img_blur[:,:,1] > blur_th
    
        seg0 = measure.label(bw1_cells)
            
        blur_th2 = skimage.filters.threshold_otsu(np.array(img_blur[~bw1_cells,1]))
        
        bw1_cells = img_blur[:,:,1] > blur_th2

        seg0 = measure.label(bw1_cells)

        propsall = measure.regionprops(seg0)#,coordinates='rc')
        out_count = 0
        for i in range(len(propsall)):
            props = propsall[i]
            if props.area < 700:
                bw = seg0 == props.label
                seg0[bw] = 0
                out_count += 1
        print(len(propsall) - out_count,edge_starting)

            
    
        ths.append(edge_starting)
#%% relabelling 
        seg_temp = seg0.copy()
        seg0[:,:] = 0
        seg0_label = 1
        added = []
        for row in range(seg_temp.shape[0]):
            for col in range(seg_temp.shape[1]):
                target = seg_temp[row,col]
                if target > 0 and target not in added:
                    bw = seg_temp == target
                    added.append(target)
                    seg0[bw] = seg0_label
                    seg0_label +=1
    
        img_out0 = label_img(img_cell,seg0)
        seg0s.append(seg0)
        cv2.imwrite(pathout+str(j).zfill(2)+'.png',img_out0)
        
        print('frame: ',j,', seg number: ',seg0_label-1, 'edge_return: ', edge_th)
        

    f = open(pathout[:-6]+'.seg0_all.pkl','wb')
    pickle.dump(seg0s,f)
    f.close()
    
    f = open(pathout[:-6]+'.ths_all.pkl','wb')
    pickle.dump(ths,f)
    f.close()
