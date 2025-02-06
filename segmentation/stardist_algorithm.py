#import celldetection as cd
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.image import imread
from skimage.transform import resize
from stardist import random_label_cmap
from stardist.models import StarDist2D
#from tifffile import imread as tifread
from csbdeep.utils import normalize
import seaborn as sns

raw_image = imread('../photo_data/1/BG-11C-6625-01-14-16-07-36.jpg')

data = resize(raw_image, (512, 512, 3))
plt.figure(figsize=(8, 8))
plt.imshow(data)
plt.show()

model = StarDist2D.from_pretrained('2D_versatile_he')


#img = data[:, :, 0]
img = data
img = normalize(img, 1,99.8, axis=(0, 1, 2))
labels, details = model.predict_instances(img, prob_thresh=0.5)

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.imshow(labels, cmap=random_label_cmap(), alpha=0.5)
plt.title('p=0.5')
plt.show()

labels, details = model.predict_instances(img, prob_thresh=0.6)

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.imshow(labels, cmap=random_label_cmap(), alpha=0.5)
plt.title('p=0.6')
plt.show()

labels, details = model.predict_instances(img, prob_thresh=0.3)

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.imshow(labels, cmap=random_label_cmap(), alpha=0.5)
plt.title('p=0.3')
plt.show()