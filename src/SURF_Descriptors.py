import pickle
import cv2
import os 
import re
import numpy as np

def extract_surf(img, hessian=50, octave=4, octaveLayers=2, ext=False):    
    
    surf = cv2.SURF(hessianThreshold=hessian, nOctaves=octave, nOctaveLayers=octaveLayers, extended=ext)
    kp, des = surf.detectAndCompute(img,None)
    
    return kp,des
    
dataDir = "../TUDarmstadt/PNGImages/sideviews-cars"
filedir = "../TUDarmstadt/Annotations/sideviews-cars"

listDir = os.listdir(dataDir)

annotedDescriptors = []

for img in listDir: 
    
    im=cv2.imread(dataDir+ "/" +img)
    
    surf = cv2.SURF(0)
    dense=cv2.FeatureDetector_create("Dense")
    dense.setInt('initXyStep', 5 )  
    #dense=cv2.DenseFeatureDetector()
    imgGray=cv2.cvtColor(im, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    kp=dense.detect(imgGray)
    #k, des = extract_surf(gray)
    k,des=surf.compute(imgGray,kp)

    
    
    with open(filedir +"/" + img.replace("png","txt"), "r") as f:
        line =  f.readlines()[-2]
        numbers =  map(int, (re.findall(r'\d+', line)))
        numbers.pop(0)
        
        t = (k, des, numbers)
        annotedDescriptors.append(t)
