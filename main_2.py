import time
import cv2
import numpy as np
import matplotlib.pyplot as plt


def max_filtering(height, width, size, I_temp):
    n = (size//2)
    wall = np.zeros((height+n*2, width+n*2), dtype=np.int32)
    temp = wall.copy()
    wall[n:height+n, n:width+n] = I_temp.copy()

    for y in range(height):
        for x in range(width):
            temp[y+n, x+n] = np.max(wall[y:y+size, x:x+size])
    return temp[n:height+n, n:width+n]


def min_filtering(height, width, size, A):
    n = (size // 2)
    wall_min = np.full((height+n*2, width+n*2), 255, dtype=np.int32)
    temp_min = wall_min.copy()
    wall_min[n:height+n, n:width+n] = A.copy()

    for y in range(height):
        for x in range(width):
            temp_min[y+n, x+n] = np.min(wall_min[y:y+size, x:x+size])
    return temp_min[n:height+n, n:width+n]


def background_subtraction(original_img, B):
    diff = original_img - B
    return cv2.normalize(diff, None, 0, 255, norm_type=cv2.NORM_MINMAX)


def min_max_filtering(original_img, size):
    height, width = original_img.shape[:2]
    max_img = max_filtering(height, width, size, original_img)
    plt.imshow(max_img)
    plt.show()
    min_img = min_filtering(height, width, size, max_img)
    plt.imshow(min_img)
    plt.show()
    return background_subtraction(original_img, min_img)


def plot_histogram(img):
    hist_img = img.astype(np.uint8)
    histogram, bin_edges = np.histogram(hist_img, bins=256)
    plt.figure()

    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here

    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.show()


if __name__ == '__main__':
    t0 = time.time()
    img = cv2.imread('datasets/009_021.jpg', cv2.IMREAD_GRAYSCALE)
    output = min_max_filtering(img, 21)
    output = output.astype(np.uint8)
    _, th3 = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    t1 = time.time()
    print(f'used time = {t1 - t0}')
    plt.imshow(th3, cmap='gray', vmin=0, vmax=255)
    plt.show()
