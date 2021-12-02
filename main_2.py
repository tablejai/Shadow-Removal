import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading


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


def background_subtraction(original_img, B, title = ""):
    # this method works on rgb image
    diff = original_img - B + 240
    
    # this method works on gray image
    # temp  = cv2.normalize(diff, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    return diff


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

def min_max_filtering(original_img, size, title = ""):
    height, width = original_img.shape[:2]
    kernel_size = 5
    kernel = np.ones((kernel_size,kernel_size),np.float32)/(kernel_size*kernel_size)

    max_img = max_filtering(height, width, size, original_img)    
    max_img_blurred = cv2.filter2D(max_img.astype(np.float32), -1, kernel)

    kernel_size = 3
    kernel = np.ones((kernel_size,kernel_size),np.float32)/(kernel_size*kernel_size)
    min_img = min_filtering(height, width, size, max_img_blurred)
    min_img_blurred = cv2.filter2D(min_img.astype(np.float32), -1, kernel)

    res = cv2.hconcat((max_img_blurred, min_img_blurred))
    cv2.imshow(title, res.astype(np.uint8))

    temp =  background_subtraction(original_img, min_img_blurred, title)
    
    # plot_histogram(temp)
    return temp


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

def foo(arg1, arg2, arg3):
    print(arg3)
    
if __name__ == '__main__':
    t0 = time.time()

    img = cv2.imread('datasets/test.jpg')
    img = cv2.resize(img, (0,0), fx=0.1, fy=0.1);

    img_b, img_g, img_r = cv2.split(img)
 
    img_b = min_max_filtering(img_b, 11, "B").astype(np.uint8)
    img_g = min_max_filtering(img_g, 11, "G").astype(np.uint8)
    img_r = min_max_filtering(img_r, 11, "R").astype(np.uint8)
    
    t1 = time.time()
    print(f'used time = {t1 - t0}')
    
    
    merged = cv2.merge((img_b, img_g, img_r))
    final_img = cv2.hconcat((img, merged))
    cv2.imshow("img", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

