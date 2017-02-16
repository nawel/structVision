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
    k, des = extract_surf(im)
    with open(filedir +"/" + img.replace("png","txt"), "r") as f:
        line =  f.readlines()[-2]
        numbers =  map(int, (re.findall(r'\d+', line)))
        numbers.pop(0)
        
        t = (des, numbers)
        annotedDescriptors.append(t)

        pickle.dump(annotedDescriptors, open("annotedDescriptors.pkl", "wb"))