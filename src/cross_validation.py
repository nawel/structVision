import numpy as np
from sklearn.model_selection import KFold


dataDir = "../dataset/TUDarmstadt/PNGImages/sideviews-cars"
filedir = "../dataset/TUDarmstadt/Annotations/sideviews-cars"

listDir = os.listdir(dataDir)

#uploading coodbook
codebook=pickle.load(open("centroids.pkl", "rb"))
annotedDescriptors = []

#creating annotated descriptors
for id_, img in zip(range(len(listDir)), listDir): 
    
    print id_, img
    im=cv2.imread(dataDir+ "/" +img)
    
    surf = cv2.SURF(0)
    dense=cv2.FeatureDetector_create("Dense")
    dense.setInt('initXyStep', 4 )  
    #dense=cv2.DenseFeatureDetector()
    imgGray=cv2.cvtColor(im, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    #kp=dense.detect(imgGray)
    #k,des=surf.compute(imgGray,kp)
    k, des = surf.detectAndCompute(imgGray,None)
    clusters, distance = vq(des,codebook)
    
    
    with open(filedir +"/" + img.replace("png","txt"), "r") as f:
        line =  f.readlines()[-2]
        numbers =  map(int, (re.findall(r'\d+', line)))
        numbers.pop(0)
        
        t = (id_, k, clusters, numbers)
        annotedDescriptors.append(t)
        
# creating Cross validation gtbox files to be used with the svmstruct command line

n_folds=5
X = annotedDescriptors
kf = KFold(n_splits=n_folds)
for (train, test), n in zip(kf.split(X), range(n_folds)):
    with open('gtbox/train_%d.gtbox'% n,'w') as f_gtbox:
            sub=[annotedDescriptors[i] for i in list(train)]
            for id_, k, clusters, numbers in sub:
                line=["%06d" % id_]+numbers   
                f_gtbox.write(' '.join(str(x) for x in line))
                f_gtbox.write('\n')
    with open('gtbox/test_%d.gtbox'% n,'w') as f_gtbox:
            sub=[annotedDescriptors[i] for i in list(test)]
            for id_, k, clusters, numbers in sub:
                line=["%06d" % id_]+numbers   
                f_gtbox.write(' '.join(str(x) for x in line))
                f_gtbox.write('\n')
                
with open('run_script.s','w') as script:
    script.write('#!/bin/bash')
    for c in [1, 0.1, 0.01, 0.001]:
        for i in range(5):
            script.write("./svm-python-v204/svm_python_learn --m subwindow -c %.3f gtbox/train_%d.gtbox output/car_%d_%.3f.model" % (c, i,i,c))
            script.write("./svm-python-v204/svm_python_classify --m subwindow gtbox/test_%d.gtbox output/car_%d_%.3f.model output/output_%d_%.3f.txt"  % (i, i, c, i,c))

