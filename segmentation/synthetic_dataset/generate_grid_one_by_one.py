import os.path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

from segmentation.synthetic_dataset.texture_filter import add_noise, texture, blank_image


def generate_dataset(n_samples=3):
    """
    Function for generation synthetic data with micro images,
    number of cells, its position and size are randomized.
    Dataset will be used for grid detection NN training
    :param n_samples: int number of images to generate
    """
    shape = (3024, 4032)
    back_val = 100

    min_cells_num = 150
    max_cells_num = 400
    min_radius = 5
    max_radius = 65

    vertical_num = 4
    vertical_w = shape[1] // vertical_num
    horizontal_num = 3
    horizontal_w = shape[0] // horizontal_num

    width_discrepancy = 30  # px
    height_discrepancy = 30  # px

    for im in range(n_samples):
        ds = np.full((shape[0], shape[1]), back_val)

        print(f'Generate {im} sample')
        v_lines = []
        h_lines = []

        # generate vertical lines
        start_v_pos = np.random.randint(20, 500)
        v_lines.append(start_v_pos)
        ds[:, start_v_pos - 15:start_v_pos + 15] = 255
        ds[:, start_v_pos - 3:start_v_pos + 3] = back_val
        for v in range(1, vertical_num):
            pos = start_v_pos + vertical_w * v + np.random.randint(-width_discrepancy, width_discrepancy)
            v_lines.append(pos)
            ds[:, pos - 15:pos + 15] = 255
            ds[:, pos - 3:pos + 3] = back_val

        start_h_pos = np.random.randint(20, 500)
        h_lines.append(start_h_pos)
        ds[start_h_pos - 15:start_h_pos + 15, :] = 255
        ds[:, start_h_pos - 3:start_h_pos + 3] = back_val
        for h in range(1, horizontal_num):
            pos = start_h_pos + horizontal_w * h + np.random.randint(-height_discrepancy, height_discrepancy)
            h_lines.append(pos)
            ds[pos - 15:pos + 15, :] = 255
            ds[pos - 3:pos + 3, :] = back_val

        # save grid positions
        w = 30
        ds_target = np.full((shape[0], shape[1]), 0)
        #ds_target_s = np.full((shape[0], shape[1]), 0)
        for l in v_lines:
            ds_target[:, l - w:l + w] = 255
            #ds_target_s[:, l - 200:l + 200] = 255
        for l in h_lines:
            ds_target[l - w:l + w, :] = 255
            #ds_target_s[l - 200:l + 200, :] = 255
        target = ds_target
        '''ds_target_s = gaussian_filter(ds_target_s, sigma=100)
        ds_target_s1 = gaussian_filter(ds_target, sigma=30)
        target = ds_target_s + ds_target_s1
        target = target / target.max()
        target[ds_target != 0] = 1'''

        '''plt.imshow(target, cmap='Grays')
        plt.colorbar()
        plt.show()'''

        ds = gaussian_filter(ds, sigma=30)

        cells_num = np.random.randint(low=min_cells_num, high=max_cells_num)
        cells_positions = np.stack((np.random.randint(low=0, high=4032, size=cells_num),
                                    np.random.randint(low=0, high=3024, size=cells_num))).T
        cells_color = np.random.randint(low=back_val + 100, high=255, size=cells_num)
        cells_radiuses = np.random.randint(low=min_radius, high=max_radius, size=cells_num)
        cells_angles = np.random.randint(low=0, high=60, size=cells_num)

        for i in range(len(cells_positions)):
            image = cv2.ellipse(img=ds,
                                center=cells_positions[i],
                                axes=(cells_radiuses[i]-4, cells_radiuses[i] + 10),
                                angle=30,
                                startAngle=0,
                                endAngle=360,
                                color=cells_color[i].astype(float),
                                thickness=-1)
            image = cv2.ellipse(img=ds,
                                center=cells_positions[i]+20,
                                axes=(cells_radiuses[i]-4, cells_radiuses[i] + 10),
                                angle=cells_angles[i],
                                startAngle=0,
                                endAngle=360,
                                color=20,
                                thickness=15)
        '''plt.imshow(image, cmap='Grays')
        plt.show()'''

        image = gaussian_filter(image, sigma=10)
        '''plt.imshow(image, cmap='Grays')
        plt.show()'''

        noise = add_noise(texture(blank_image(background=230), sigma=8), sigma=10)[:, :, 0]
        noise = resize(noise, (3024, 4032))
        image = image + noise
        '''plt.imshow(image, cmap='Grays')
        plt.show()'''


        target_size = (256, 256)
        image = resize(image, (
                         target_size[0],
                         target_size[1]))
        print('resize target')
        target = resize(target, (
                                       target_size[0],
                                       target_size[1]))

        image = (image - image.min()) / (image.max() - image.min())
        target = (target - target.min()) / (target.max() - target.min())
        '''plt.imshow(image, cmap='Grays')
        plt.colorbar()
        plt.show()
        plt.imshow(target, cmap='Grays')
        plt.colorbar()
        plt.show()'''

        folder = f'synthetic_dataset/{image.shape[0]}_{image.shape[1]}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(f'{folder}/img{im}.npy', image)
        np.save(f'{folder}/target{im}.npy', target)

#generate_dataset(50)

'''folder = 'synthetic_dataset/151_201'
for file in os.listdir(folder):
    if 'img' in file:
        f = np.load(f'{folder}/{file}')
        t = np.load(f'{folder}/{file.replace("img", "target")}')
        plt.imshow(f, cmap='Grays')
        plt.tight_layout()
        plt.show()'''

