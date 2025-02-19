import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.transform import resize
from torch import float32

from segmentation.grid_segmentation.unet_mini import UNetMini

device = 'cuda'

"""
Script for making prediction - segment grid with trained UnetMini on real image 
"""


model = UNetMini(num_classes=2).to(device)
model.load_state_dict(torch.load('CEL_unet/best_unet.pt', weights_only=True))

img = cv2.imread('../photo_data/1/BG-11C-6625-01-14-16-08-54.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255

gray = np.expand_dims(resize(gray, (256, 256)), axis=0)
gray = np.expand_dims(gray, axis=0)

gray = torch.tensor(gray, dtype=float32).to(device)

pred = model(gray)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
cs = axs[0].imshow(pred.cpu().detach().numpy()[0][0], vmax=20)
plt.colorbar(cs, ax=axs[0])
axs[0].set_title('Prediction')
cs = axs[1].imshow(gray.cpu().detach().numpy()[0][0])
plt.colorbar(cs, ax=axs[1])
axs[1].set_title('Target')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
cs = axs[0].imshow(pred.cpu().detach().numpy()[0][1], vmin=-20)
plt.colorbar(cs, ax=axs[0])
axs[0].set_title('Prediction')
cs = axs[1].imshow(gray.cpu().detach().numpy()[0][0])
plt.colorbar(cs, ax=axs[1])
axs[1].set_title('Target')
plt.tight_layout()
plt.show()






