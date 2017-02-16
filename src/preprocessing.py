import pickle
import cv2
import os 
import re
import numpy as np
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from itertools import izip as zip, count

def extract_surf(img, hessian=50, octave=4, octaveLayers=2, ext=False):    
    
    surf = cv2.SURF(hessianThreshold=hessian, nOctaves=octave, nOctaveLayers=octaveLayers, extended=ext)
    kp, des = surf.detectAndCompute(img,None)
    
    return kp,des
    
dataDir = "../dataset/TUDarmstadt/PNGImages/sideviews-cars"
filedir = "../dataset/TUDarmstadt/Annotations/sideviews-cars"

listDir = os.listdir(dataDir)

codebook=pickle.load(open("centroids.pkl", "rb"))
annotedDescriptors = []

for id_, img in zip(range(len(listDir)), listDir): 
    
    print id_, img
    im=cv2.imread(dataDir+ "/" +img)
    
    surf = cv2.SURF()
    dense=cv2.FeatureDetector_create("Dense")
    dense.setInt('initXyStep', 5 )  
    #dense=cv2.DenseFeatureDetector()
    imgGray=cv2.cvtColor(im, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    kp=dense.detect(imgGray)
    #k, des = extract_surf(gray)
    k,des=surf.compute(imgGray,kp)

    clusters, distance = vq(des,codebook)
    
    
    with open(filedir +"/" + img.replace("png","txt"), "r") as f:
        line =  f.readlines()[-2]
        numbers =  map(int, (re.findall(r'\d+', line)))
        numbers.pop(0)
        
        t = (id_, k, clusters, numbers)
        annotedDescriptors.append(t)

#pickle.dump(annotedDescriptors, open("annotedDescriptors.pkl", "wb"))


##################### CREATING the preprocessed files 

train_dir='train/'
gtbox_path='car.gtbox'

#if train directory doesn't exist, create it
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

#create the gtbox file
with open(gtbox_path,'w') as f_gtbox:
    for id_, k, clusters, numbers in annotedDescriptors:
        
        line=["%06d" % id_]+numbers  
        f_gtbox.write(' '.join(str(x) for x in line))
        f_gtbox.write('\n')
        lines_clst=[]
        for kp, cluster in zip(k,clusters):
            line_clst="%d %d %d" % (kp.pt[0],kp.pt[1],cluster)
            lines_clst.append(line_clst)
        #create the clst file for each image
        with open(train_dir+'%06d.clst' % id_, 'w') as f:
            f.write('\n'.join(lines_clst))
