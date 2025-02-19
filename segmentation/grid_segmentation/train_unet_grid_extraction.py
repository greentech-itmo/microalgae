import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

from segmentation.grid_segmentation.unet_mini import UNetMini

device = 'cuda'

"""
Script for training UnetMini for grid segmentation
Validation on real synthetic data
"""


model = UNetMini(num_classes=2).to(device)

print(summary(model, input_size=(1, 256, 256)))


exp_name = 'CEL_unet1'
if not os.path.exists(f'grid_segmentation/{exp_name}'):
    os.makedirs(f'grid_segmentation/{exp_name}')


def load_data(folder):
    features = []
    target = []

    for file in os.listdir(folder):
        if 'img' in file:
            f = np.load(f'{folder}/{file}')
            t = np.load(f'{folder}/{file.replace("img", "target")}')
            features.append(f)
            target.append(t)
    features = np.array(features)
    target = np.array(target)
    return features, target


features, target = load_data('synthetic_dataset/256_256')

features = np.expand_dims(features, axis=1)
#target = np.expand_dims(target, axis=1)
target[target < 0.1] = 0
target[target >= 0.1] = 1

reverse_target = np.full(target.shape, 0)
reverse_target[target == 0] = 1

target = np.stack([target, reverse_target], axis=1)

plt.imshow(features[5][0])
plt.colorbar()
plt.show()
plt.imshow(target[5][0])
plt.colorbar()
plt.show()

print('Form dataset')
size = features.shape[0]

split = (0.1, 0.1, 0.2)
train_s = int(split[0] * size)
val_s = int(split[1] * size)
test_s = int(split[2] * size)

features_train = torch.tensor(features[:train_s].astype(np.float32))
target_train = torch.tensor(target[:train_s].astype(np.float32))

features_val = torch.tensor(features[train_s:train_s + val_s].astype(np.float32))
target_val = torch.tensor(target[train_s:train_s + val_s].astype(np.float32))

features_test = torch.tensor(features[train_s + val_s:train_s + val_s + test_s].astype(np.float32))
target_test = torch.tensor(target[train_s + val_s:train_s + val_s + test_s].astype(np.float32))

features = None
target = None

train_ds = TensorDataset(features_train, target_train)
val_ds = TensorDataset(features_val, target_val)
print('Form loader')
train_loader = DataLoader(train_ds, batch_size=300, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=300, shuffle=True)

optim = torch.optim.AdamW(params=model.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()

epochs = 100000
losses = []
val_loses = []
best_model = None
best_loss = np.inf
best_epoch = None

for ep in range(epochs):
    loss = 0
    val_loss = 0
    for features, target in train_loader:
        features = features.to(device)
        target = target.to(device)
        pred = model(features)
        batch_loss = criterion(pred, target)
        batch_loss.backward()
        optim.step()
        loss += batch_loss.item()

    for v_features, v_target in val_loader:
        v_features = v_features.to(device)
        v_target = v_target.to(device)
        pred = model(v_features)
        batch_loss = criterion(pred, v_target)
        val_loss += batch_loss.item()

    loss = loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    losses.append(loss)
    val_loses.append(val_loss)

    print(f'epoch {ep}/{epochs}, loss={loss}, validation={val_loss}')

    if val_loses[-1] < best_loss:
        torch.save(model.state_dict(), f'grid_segmentation/{exp_name}/best_unet.pt')
        best_model = model
        best_loss = val_loses[-1]
        best_epoch = ep

    if ep % 500 == 0 or ep == epochs-1:

        prediction = best_model(v_features).cpu().detach().numpy()
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        cs = axs[0].imshow(prediction[0][0])
        plt.colorbar(cs, ax=axs[0])
        axs[0].set_title('Prediction')
        cs = axs[1].imshow(v_target.cpu().detach().numpy()[0][0])
        plt.colorbar(cs, ax=axs[1])
        axs[1].set_title('Target')
        plt.suptitle(f'Target1 Epoch {ep}')
        plt.tight_layout()
        plt.savefig(f'grid_segmentation/{exp_name}/{ep}_pred1.png')
        plt.close()

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        cs = axs[0].imshow(prediction[0][1])
        plt.colorbar(cs, ax=axs[0])
        axs[0].set_title('Prediction')
        cs = axs[1].imshow(v_target.cpu().detach().numpy()[0][1])
        plt.colorbar(cs, ax=axs[1])
        axs[1].set_title('Target')
        plt.suptitle(f'Target2 Epoch {ep}')
        plt.tight_layout()
        plt.savefig(f'grid_segmentation/{exp_name}/{ep}_pred2.png')
        plt.close()

        plt.plot(np.arange(len(losses)), losses, label='Train')
        plt.plot(np.arange(len(losses)), val_loses, label='Validation')
        plt.title('Convergence plot')
        plt.axhline(best_loss, c='green', linestyle='dashed')
        plt.annotate(str(round(best_loss, 4)), (0, best_loss), c='green')
        plt.axvline(best_epoch, c='green',linestyle='dashed')
        plt.annotate(str(int(best_epoch)), (best_epoch, losses[0]), c='green')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'grid_segmentation/{exp_name}/unet_convergence.png')
        plt.show()
