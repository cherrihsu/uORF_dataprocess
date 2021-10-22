# -*- coding: utf-8 -*-
"""
seg1 tracking
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
    propsall = measure.regionprops(seg,coordinates = 'rc')
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


#%% cell numbers
path = './data/20210928 90h Lf/Se48Lf7/'
position = 25

cell_path = path + 'MAX_Position ' + str(position).zfill(2) +' - C=0.seg1/'

tifs  = []
seg1_file = []
ths_file = []

tifs.append( path+'MAX_Position ' + str(position).zfill(2) +' - C=0.tif')
seg1_file.append(path+'MAX_Position ' + str(position).zfill(2) +'.seg1_all.pkl')
ths_file.append(path+'MAX_Position ' + str(position).zfill(2) +'.ths_all.pkl')

im = io.imread(tifs[0])
print(tifs[0])
print(seg1_file[0])

segfile = seg1_file[0]
f = open(segfile,'rb')
seg1s = pickle.load(f)
f.close()

segfile = ths_file[0]
f = open(segfile,'rb')
ths = pickle.load(f)
f.close()

pathout = tifs[0][:-10]+'.seg2/'
try:
    os.mkdir(pathout)
except:
    print('no dir made')

seg2s = []
seg2 = seg1s[0].copy()


propsall = measure.regionprops(seg2,coordinates = 'rc')
for prop_index in range(len(propsall)):
    props = propsall[prop_index]
    ratio = props.major_axis_length / props.minor_axis_length

    if props.area > 5400 or ratio > 1.5:
        for next_index in range(10): 
            seg2_next = seg1s[next_index+1].copy()
            bw0 = seg2 == props.label
            labels = np.unique( seg2_next[bw0] )
            counts = 0
            for label in labels:
                if label > 0 and np.sum(seg2_next[bw0] == label) > 400:
                    counts +=1
            if counts > 1:
                bw0 = seg2 == props.label
                seg2[bw0] = 0
                for label in labels:
                    if label > 0:
                        bw1 = seg2_next == label
                        bw2 = bw0 & bw1
                        if np.sum(bw2) > 400:
                            seg2[bw2] = np.max(seg2)+1
                break

seg_temp = seg2.copy()
seg2[:,:] = 0
seg2_label = 1
added = []

for row in range(seg_temp.shape[0]):
    for col in range(seg_temp.shape[1]):
        target = seg_temp[row,col]
        if target > 0 and target not in added:
            bw = seg_temp == target
            bw = ndimage.binary_fill_holes(bw).astype(bool)
            added.append(target)
            seg2[bw] = seg2_label
            seg2_label +=1    


img0 = im[0]
edge_th = ths[0]
[edge_th,img_cell] = cell_edge(img0,edge_th)
img_out0 = label_img(img_cell,seg2)
cv2.imwrite(pathout+'seg2.'+str(0).zfill(2)+'.png',img_out0)
seg2s.append(seg2.copy())

#%% start tracking
for frame_index in range(1,len(seg1s)):
    
    seg1 = seg1s[frame_index].copy()
    seg_bad = seg1.copy()
    seg_bad[:,:] = 0
    edge_th = ths[frame_index]

    propsall = measure.regionprops(seg1,coordinates = 'rc')
    for prop_index in range(len(propsall)):
        props = propsall[prop_index]
        ratio = props.major_axis_length / props.minor_axis_length
        if props.area > 5400 or ratio > 1.5:
            bw = seg1==props.label
            seg1[bw] = 0
            seg_bad[bw] = props.label

    seg2_old = seg2.copy()
    seg2[:,:] = 0        

    propsall = measure.regionprops(seg2_old,coordinates = 'rc')
    print(len(propsall),' tracks in frame',frame_index-1)
    for i in range(len(propsall)):
        props = propsall[i]
        bw0 = seg2_old== props.label
        
        labels_now = list(np.unique(seg1[bw0])) # maping seg2_old to seg1
        if 0 in labels_now:
            labels_now.remove(0)
        
        track_label = 0
        if len(labels_now) == 1: # maybe [0, label] or [label]
            if labels_now[-1] > 0:
                bw1 = seg1 == labels_now[-1]
                    
                if np.sum(bw0 & bw1) > 300: 
                    if np.sum(bw1) > 2.5* np.sum(bw0):
                        bw2 = bw0 & bw1
                        track_label = labels_now[-1]
                        seg2[bw2] = props.label
                        seg1[bw2] = 0
                    else:
                        track_label = labels_now[-1]
                        seg2[bw1] = props.label
                        seg1[bw1] = 0                    
        else: # find biggest size
            max_size = 0
            max_label = 0
            for label in labels_now:
                area = np.sum(seg1[bw0] == label)
                if  area > max_size and area > 400:
                    max_size = area
                    max_label = label
            if max_label > 0:
                track_label = max_label
                bw1 = seg1 == track_label
                seg2[bw1] = props.label
                seg1[bw1] = 0                  
        if track_label == 0: # no further tracks
            
            label_bad = np.unique(seg_bad[bw0])
            if len(label_bad) <=2:
                if label_bad[-1] > 0:
                    track_label = label_bad[-1]
                    bw2 = seg_bad == track_label
                    bw3 = bw2 & bw0
                    seg2[bw3] = props.label
                    seg1[bw3] = 0
            else:
                max_size = 0
                max_label = 0
                for label_index in range(1,len(label_bad)):
                    area = np.sum(seg_bad[bw0] == label_bad[label_index])
                    if  area > max_size:
                        max_size = area
                        max_label = label_bad[label_index]
                if max_label > 0:
                    track_label = max_label
                    bw3 = seg_bad == track_label
                    seg2[ bw3 & bw0] = props.label
                    seg_bad[bw3 & bw0] = 0 
        if np.sum(seg2 == props.label) < 400:
            seg2[seg2 == props.label] = 0
   
    img0 = im[frame_index]
    [edge_th,img_cell] = cell_edge(img0,edge_th)        
    img_out0 = label_img(img_cell,seg2)
    seg2s.append(seg2.copy())
    cv2.imwrite(pathout+'seg2.'+str(frame_index).zfill(2)+'.png',img_out0)
    old_labels = np.unique(seg2s[-2])
    labels = np.unique(seg2s[-1])
#%%
propsall = measure.regionprops(seg2,coordinates = 'rc')
print(len(propsall),' tracks in frame', len(seg2s)-1)
f = open(pathout[:-6]+'.seg2_all.pkl','wb')
pickle.dump(seg2s,f)
f.close()
