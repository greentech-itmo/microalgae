import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

from segmentation.synthetic_dataset.texture_filter import add_noise, texture, blank_image


def get_real_stats():
    img = cv2.imread('../photo_data/1/BG-11C-6625-01-14-16-08-54.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='Grays')
    plt.show()
    return np.median(gray)


def generate_dataset(n, n_samples=3):
    back_val = get_real_stats()
    back_val = 100
    ds = np.full((n_samples, 3024, 4032), back_val)
    ds_target = np.full((n_samples, 3024, 4032), 0)

    min_cells_num = 150
    max_cells_num = 400
    min_radius = 5
    max_radius = 65

    vertical_num = 4
    vertical_w = ds.shape[2] // vertical_num
    horizontal_num = 3
    horizontal_w = ds.shape[1] // horizontal_num

    width_discrepancy = 30  # px
    height_discrepancy = 30  # px

    for im in range(n_samples):
        print(f'Generate {im} sample')
        v_lines = []
        h_lines = []

        # generate vertical lines
        start_v_pos = np.random.randint(20, 500)
        v_lines.append(start_v_pos)
        ds[im, :, start_v_pos - 15:start_v_pos + 15] = 255
        ds[im, :, start_v_pos - 3:start_v_pos + 3] = back_val
        for v in range(1, vertical_num):
            pos = start_v_pos + vertical_w * v + np.random.randint(-width_discrepancy, width_discrepancy)
            v_lines.append(pos)
            ds[im, :, pos - 15:pos + 15] = 255
            ds[im, :, pos - 3:pos + 3] = back_val

        start_h_pos = np.random.randint(20, 500)
        h_lines.append(start_h_pos)
        ds[im, start_h_pos - 15:start_h_pos + 15, :] = 255
        ds[im, :, start_h_pos - 3:start_h_pos + 3] = back_val
        for h in range(1, horizontal_num):
            pos = start_h_pos + horizontal_w * h + np.random.randint(-height_discrepancy, height_discrepancy)
            h_lines.append(pos)
            ds[im, pos - 15:pos + 15, :] = 255
            ds[im, pos - 3:pos + 3, :] = back_val

        # save grid positions


        for l in v_lines:
            ds_target[im, :, l-200:l+200] = 255
        for l in h_lines:
            ds_target[im, l-200:l+200, :] = 255
        '''plt.imshow(ds_target[im])
        plt.show()
        plt.imshow(ds[im])
        plt.show()'''

    ds_target_s = gaussian_filter(ds_target, sigma=200, axes=(1, 2))
    ds_target = ds_target+ds_target_s
    ds_target = ds_target/ds_target.max()
    ds = gaussian_filter(ds, sigma=30, axes=(1, 2))

    cells_num = np.random.randint(low=min_cells_num, high=max_cells_num, size=n_samples)
    cells_positions = [np.stack((np.random.randint(low=0, high=4032, size=num),
                                 np.random.randint(low=0, high=3024, size=num))).T for num in cells_num]
    cells_color = [np.random.randint(low=back_val+100, high=255, size=num) for num in cells_num]
    cells_radiuses = [np.random.randint(low=min_radius, high=max_radius, size=num) for num in cells_num]
    cells_angles = [np.random.randint(low=0, high=60, size=num) for num in cells_num]


    for im in range(ds.shape[0]):
        for i in range(len(cells_positions[im])):
            image = cv2.ellipse(img=ds[im],
                                center=cells_positions[im][i],
                                axes=(cells_radiuses[im][i]-4, cells_radiuses[im][i] + 10),
                                angle=30,
                                startAngle=0,
                                endAngle=360,
                                color=cells_color[im][i].astype(float),
                                thickness=-1)
            image = cv2.ellipse(img=ds[im],
                                center=cells_positions[im][i]+20,
                                axes=(cells_radiuses[im][i]-4, cells_radiuses[im][i] + 10),
                                angle=cells_angles[im][i],
                                startAngle=0,
                                endAngle=360,
                                color=20,
                                thickness=15)
        #plt.imshow(image, cmap='Grays')
        #plt.show()

        image = gaussian_filter(image, sigma=10)
        #plt.imshow(image, cmap='Grays')
        #plt.show()

        noise = add_noise(texture(blank_image(background=230), sigma=8), sigma=10)[:, :, 0]
        noise = resize(noise, (3024, 4032))

        #plt.imshow((image+noise)/(image+noise).max()*255, cmap='Grays')
        #plt.show()

        ds[im] = image+noise

    print('resize ds')

    scale = 10

    ds = resize(ds, (ds.shape[0],
                     ds.shape[1]//scale,
                     ds.shape[2]//scale))
    ds = ds/ds.max()
    print('resize target')
    ds_target = resize(ds_target, (ds_target.shape[0],
                                   ds_target.shape[1] // scale,
                                   ds_target.shape[2] // scale))
    ds_target = ds_target / ds_target.max()

    plt.imshow(ds[0], cmap='Grays')
    plt.colorbar()
    plt.show()
    plt.imshow(ds_target[0], cmap='Grays')
    plt.colorbar()
    plt.show()

    np.save(f'synthetic_dataset/images{n}_{ds[1]}_{ds[1]}.npy', ds)
    np.save(f'synthetic_dataset/grid_mask{n}_{ds[1]}_{ds[1]}.npy', ds_target)


for i in range(1, 5):
    generate_dataset(i, n_samples=5)
