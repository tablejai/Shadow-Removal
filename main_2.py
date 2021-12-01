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


def ROI_mean(img, mask):
    sum_ = 0
    cnt = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i, j]:
                cnt = cnt + 1
                sum_ += img[i][j]

    mean_ = sum_ / cnt
    print(mean_)
    return mean_


def min_max_filtering(original_img, size):
    height, width = original_img.shape[:2]
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.float32) / \
        (kernel_size*kernel_size)

    max_img = max_filtering(height, width, size, original_img)
    max_img_blurred = cv2.filter2D(max_img.astype(np.float32), -1, kernel)

    min_img = min_filtering(height, width, size, max_img_blurred)
    min_img_blurred = cv2.filter2D(min_img.astype(np.float32), -1, kernel)

    _, shadow_filter = cv2.threshold(max_img_blurred.astype(
        np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ROI_mean(original_img, 255-shadow_filter)

    plt.subplot(2, 2, 1)
    plt.imshow(max_img_blurred, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(min_img_blurred, cmap='gray')
    return background_subtraction(original_img, min_img_blurred)


def plot_histogram_rgb(img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def plot_grayscale_histogram(img):
    # density=False would make counts
    plt.subplot(2, 2, 3)
    plt.hist(np.ravel(img), density=True, bins=256)
    plt.ylabel('Probability')
    plt.xlabel('Data')


if __name__ == '__main__':
    t0 = time.time()
    plt.figure(figsize=(5, 5))

    img = cv2.imread('datasets/002_022.jpg', cv2.IMREAD_GRAYSCALE)
    output = min_max_filtering(img, 11).astype(np.uint8)

    real_text_color = np.min(output)

    _, binary = cv2.threshold(output, 150, 255, cv2.THRESH_BINARY)

    print("Binary.shape: {binary.shape}")

    fixed_output = np.ma.array(output, mask=np.bitwise_not(
        binary)).filled(fill_value=real_text_color)

    t1 = time.time()
    print(f'used time = {t1 - t0}')

    plt.subplot(2, 2, 3)
    plt.imshow(img, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.imshow(fixed_output, cmap='gray', vmin=0, vmax=255)
    plt.show()
