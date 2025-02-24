import os
import random

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage


def get_rotated_dataset(features_path, target_path, repeat=5):
    """
    Function to rotate each sample on angle - data augmentation
    :param features_path: str to npy features file
    :param target_path: str to npy target file
    :param repeat: number of random rotations to each sample
    :return: augmented features and target
    """
    median_val = 0.6
    features = np.load(features_path)
    target = np.load(target_path)

    rotated_features = []
    rotated_target = []
    for i in range(repeat):
        rot_features = []
        rot_target = []
        for t in range(features.shape[0]):
            print(f' {i + 1}: sample {t}/{features.shape[0]}')
            angle = random.uniform(-5, 5)
            rot_features.append(ndimage.rotate(features[t], angle, reshape=False, cval=median_val))
            rot_target.append(ndimage.rotate(target[t], angle, reshape=False, cval=0))
        rotated_features.extend(rot_features)
        rotated_target.extend(rot_target)

    rotated_features.extend(features.tolist())
    rotated_target.extend(target.tolist())

    return rotated_features, rotated_target


def load_full_dataset(path_to_data):
    """
    Function to load dataset from separated files and augment it
    :param path_to_data: str folder with files
    :return: full dataset with augmentation - features, target
    """
    features = []
    target = []
    for ds_num in range(1, 5):
        features_s, target_s = get_rotated_dataset(f'{path_to_data}/images{ds_num}.npy',
                                                   f'{path_to_data}/grid_mask{ds_num}.npy',
                                                   repeat=3)
        features.extend(features_s)
        target.extend(target_s)

    features = np.array(features)
    target = np.array(target)

    np.save('synthetic_dataset/images_aug.npy', features.astype(np.float32))
    np.save('synthetic_dataset/grid_mask_aug.npy', target.astype(np.float32))


def augment_files_in_folder(folder, rep_num=2):
    median_val = 0.7  # value of background to fill gaps after rotation

    samples_num = len(os.listdir(folder))//2
    for i in range(samples_num):
        img = np.load(f'{folder}/img{i}.npy')
        target = np.load(f'{folder}/target{i}.npy')

        for r in range(rep_num):
            print(f'  sample {i}/{samples_num} - rot {r+1}/{rep_num}')
            angle = random.uniform(-4, 4)
            aug_img = ndimage.rotate(img, angle, reshape=False, cval=median_val)
            aug_target = ndimage.rotate(target, angle, reshape=False, cval=0.1)

            '''plt.imshow(aug_img, cmap='Grays')
            plt.show()
            plt.imshow(aug_target, cmap='Grays')
            plt.show()'''

            np.save(f'{folder}/img{i}_{r}.npy', aug_img)
            np.save(f'{folder}/target{i}_{r}.npy', aug_target)


augment_files_in_folder('256_256')
