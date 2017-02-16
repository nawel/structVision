import pickle
import cv2
import os 
import re
import numpy as np
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler


dataDir = "../dataset/TUDarmstadt/PNGImages/sideviews-cars"
filedir = "../dataset/TUDarmstadt/Annotations/sideviews-cars"

listDir = os.listdir(dataDir)

des_list = []

# get all descriptors from all images 

for img in listDir: 
    
    image_path=dataDir+ "/" +img
    
    im=cv2.imread(image_path)
    
    surf = cv2.SURF()
    dense=cv2.FeatureDetector_create("Dense")
    dense.setInt('initXyStep', 7 )  
    #dense=cv2.DenseFeatureDetector()
    imgGray=cv2.cvtColor(im, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    kp=dense.detect(imgGray)
    #k, des = extract_surf(gray)
    k,des=surf.compute(imgGray,kp)

    des_list.append((image_path, des)) 
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor)) 

# do a kmeans clustering and save
k = 5000
codebook, variance = kmeans(descriptors, k, iter=1) 
pickle.dump(codebook, open("centroids.pkl", "wb"))
