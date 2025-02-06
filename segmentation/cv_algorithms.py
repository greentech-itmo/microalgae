import numpy as np
import cv2
from matplotlib import pyplot as plt


def detect_cell_cv2(image_path, increase_channel=None):
    img = cv2.imread(image_path)
    chnls = cv2.split(img)

    if increase_channel is not None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(chnls[increase_channel])
        new_chs = []
        for i, c in enumerate(chnls):
            if i != increase_channel:
                new_chs.append(c)
            else:
                new_chs.append(cl)
        img = cv2.merge(new_chs)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)

    minDist = 100
    param1 = 30
    param2 = 30  #smaller value-> more false circles
    minRadius = 5
    maxRadius = 100

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)

    plt.rcParams['figure.figsize'] = (8, 6)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 5)

    cells_area = np.round(np.sum(circles[0, :, 2] ** 2 * np.pi), 3)
    cells_mean_rad = int(np.round(np.mean(circles[0, :, 2])))
    plt.title(f'Full image cell number = {circles.shape[1]}\n'
              f'full cells area = {cells_area}\n'
              f'mean cell radius = {cells_mean_rad}')

    inds = np.arange(circles.shape[1])
    for ind in inds:
        plt.annotate(ind, (circles[0, ind, 0], circles[0, ind, 1]), c='blue', fontsize=9)
    plt.imshow(img)
    plt.tight_layout()
    plt.show()


def detect_edges(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='Grays')
    plt.show()

    #gray = cv2.medianBlur(gray, 25)
    edges = cv2.Canny(gray, 50, 150, apertureSize=7)

    plt.imshow(edges, cmap='Greys_r')
    plt.show()
    lines = cv2.HoughLinesP(image=gray, rho=1, theta=np.pi / 180, threshold=1000, lines=np.array([]),
                            minLineLength=500, maxLineGap=10)

    a, b, c = lines.shape
    plt.imshow(gray)
    for i in range(a):
        plt.plot([lines[i][0][0], lines[i][0][2]], [lines[i][0][1], lines[i][0][3]])
        #cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 10, cv2.LINE_AA)
    plt.show()



detect_edges('../photo_data/1/BG-11C-6625-01-14-16-08-54.jpg')
