import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

dataDir = "../dataset/TUDarmstadt/PNGImages/sideviews-cars"

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
        
im = np.array(Image.open(dataDir +'/car-pic172-sml2.png'), dtype=np.uint8)

_,t,l,b,r=pred_boxes[3]

# Create figure and axes
fig,ax = plt.subplots(1)
ax.imshow(im)
rect = patches.Rectangle((t,l),b-t,r-l,linewidth=2,edgecolor='r',facecolor='none')
ax.add_patch(rect)
_,t,l,b,r=gt_boxes[3]
rect = patches.Rectangle((t,l),b-t,r-l,linewidth=2,edgecolor='g',facecolor='none')
ax.add_patch(rect)
plt.savefig('predicted3.jpg')

