import pickle
import cv2
import os 
import re
import numpy as np
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from itertools import izip as zip, count
import matplotlib.pyplot as plt


pred_boxes=[]
with open("output/output_3_0.001.txt", "r") as f:
    lines =  f.readlines()
    for line in lines:
        pred_boxes.append(map(float, (re.findall(r'[-+]?\d*\.\d+|\d+', line))))
gt_boxes=[]
with open("gtbox/test_3.gtbox", "r") as f:
    lines =  f.readlines()
    for line in lines:
        gt_boxes.append(map(float, (re.findall(r'[-+]?\d*\.\d+|\d+', line))))
        
#ID   left top right bottom
#0     1    2    3    4

def box_area(box):
        return (box[3]-box[1]+1)*(box[4]-box[2]+1)
def box_overlap(box1,box2):
        intersection = [0,max(box1[1],box2[1]),max(box1[2],box2[2]),min(box1[3],box2[3]),min(box1[4],box2[4])]
        if (intersection[1]<=intersection[3]) & (intersection[2]<=intersection[4]):
            return box_area(intersection)/float(box_area(box1)+box_area(box2)-box_area(intersection))
        else:
            return 0.
box_overlaps=[]
npos=len(pred_boxes)
tp=np.zeros(npos)
fp=np.zeros(npos)
fn=np.zeros(npos)
for pbox, gbox, i in zip(pred_boxes,gt_boxes, range(npos)):
    overlap=box_overlap(pbox,gbox)
    if(overlap>=0.5):
        tp[i]=1
    elif(overlap<0.5) & (overlap>0):
        fp[i]=1
    else:
        fn[i]=1
fn=np.cumsum(fn)
tp=np.cumsum(tp)
fp=np.cumsum(fp)        

recall=tp/(tp+fn)
precision=tp/(fp+tp)

print recall
print precision

import pylab as pl
pl.clf()
g=[i for i in range(len(precision))]
pl.plot(g, list(precision), label='Precision curve')
pl.ylabel('Precision')
pl.xlabel('Number of examples')
pl.savefig('output.jpg')
