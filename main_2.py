import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from plot_helper import plot_histogram_gray, plot_histogram_rgb


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


def background_subtraction(original_img, background, title=""):
    diff = original_img - background
    return diff


def global_background_mean(img, mask):
    t = cv2.mean(img, mask)[0]
    cv2.imshow("img", img)
    cv2.imshow("mask", mask)
    print("gloabl mean", t)
    return t


def normalize(orig_img, img, title=""):

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title(title+"_original")

    y_hist, x_hist, _ = plt.hist(np.ravel(orig_img), density=True, bins=256)

    offset = x_hist[np.where(y_hist == y_hist.max())]
    img = img + offset

    plt.subplot(1, 2, 2)
    plt.title(title+"_normalized")
    plt.hist(np.ravel(img), density=True, bins=256)

    return img


def min_max_filtering(original_img, size, title=""):
    height, width = original_img.shape[:2]

    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.float32) / \
        (kernel_size*kernel_size)
    max_img = max_filtering(height, width, size, original_img)
    max_img_blurred = cv2.filter2D(max_img.astype(np.float32), -1, kernel)

    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.float32) / \
        (kernel_size*kernel_size)
    min_img = min_filtering(height, width, size, max_img_blurred)
    min_img_blurred = cv2.filter2D(min_img.astype(np.float32), -1, kernel)

    cv2.imshow(title, cv2.hconcat(
        (max_img_blurred, min_img_blurred)).astype(np.uint8))

    diff = background_subtraction(original_img, min_img_blurred)

    normed = normalize(original_img, diff, title)

    return normed


if __name__ == '__main__':

    img = cv2.imread('datasets/test2.jpg')
    # img = cv2.imread('datasets/001_007.jpg')
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    t0 = time.time()

    img_b, img_g, img_r = cv2.split(img)

    img_b = min_max_filtering(img_b, 11, "B").astype(np.uint8)
    img_g = min_max_filtering(img_g, 11, "G").astype(np.uint8)
    img_r = min_max_filtering(img_r, 11, "R").astype(np.uint8)

    print(f'used time = {time.time() - t0}')

    merged = cv2.merge((img_b, img_g, img_r))
    final_img = cv2.hconcat((img, merged))
    cv2.imshow("result", final_img)
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
