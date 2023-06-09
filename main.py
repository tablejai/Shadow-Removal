import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from plot_helper import plot_histogram_1Channel, plot_histogram_rgb
from skimage.metrics import structural_similarity as ssim

def max_filtering(height, width, kernel_size, input_img):
    n = (kernel_size//2)
    wall = np.zeros((height+n*2, width+n*2), dtype=np.int32)
    result = wall.copy()
    wall[n:height+n, n:width+n] = input_img.copy()

    for y in range(height):
        for x in range(width):
            result[y+n, x+n] = np.max(wall[y:y+kernel_size, x:x+kernel_size])
    return result[n:height+n, n:width+n]


def min_filtering(height, width, kernel_size, input_img):
    n = (kernel_size // 2)
    wall = np.full((height+n*2, width+n*2), 255, dtype=np.int32)
    result = wall.copy()
    wall[n:height+n, n:width+n] = input_img.copy()

    for y in range(height):
        for x in range(width):
            result[y+n, x+n] = np.min(wall[y:y+kernel_size, x:x+kernel_size])
    return result[n:height+n, n:width+n]


def background_subtraction(original_img, background, title=""):
    diff = original_img - background
    return diff


def global_background_mean(img, mask):
    t = cv2.mean(img, mask)[0]
    cv2.imshow("img", img)
    cv2.imshow("mask", mask)
    print("gloabl mean", t)
    return t

def hist_normalize(orig_img, img, title=""):
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.xlim([0, 256])
    # plt.title(title+"_original")

    y_hist, x_hist, _ = plt.hist(np.ravel(orig_img), density=True, bins=256)

    # plt.subplot(1, 3, 2)
    # plt.hist(np.ravel(img), density=True, bins=256)
    # plt.title(title+"_subtracted")

    offset = x_hist[np.where(y_hist == y_hist.max())]
    img = img + offset
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < 0:
                img[i, j] = 0
                
    # plt.subplot(1, 3, 3)
    # plt.xlim([0, 256])
    # plt.title(title+"_normalized")
    # plt.hist(np.ravel(img), density=True, bins=256)

    return img


def min_max_filtering(original_img, size, title=""):
    height, width = original_img.shape[:2]

    # max filter
    max_img = max_filtering(height, width, size, original_img)
    
    # mean filter
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    max_img_blurred = cv2.filter2D(max_img.astype(np.float32), -1, kernel)

    # min filter
    min_img = min_filtering(height, width, size, max_img_blurred)

    # mean filter
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    min_img_blurred = cv2.filter2D(min_img.astype(np.float32), -1, kernel)

    # cv2.imshow(title, cv2.hconcat((max_img_blurred, min_img_blurred)).astype(np.uint8))
    diff = background_subtraction(original_img, min_img_blurred)

    normed = hist_normalize(original_img, diff, title)
    return normed


if __name__ == '__main__':
    image_name = "test10.jpg"
    resize_factor = 1
    original_img = cv2.imread(f'datasets/self_doc_img/{image_name}')
    original_img = cv2.resize(original_img, (0, 0), fx=resize_factor, fy=resize_factor)

    t0 = time.time()

    img_b, img_g, img_r = cv2.split(original_img)
    
    img_b = min_max_filtering(img_b, 11, "B").astype(np.uint8)
    img_g = min_max_filtering(img_g, 11, "G").astype(np.uint8)
    img_r = min_max_filtering(img_r, 11, "R").astype(np.uint8)

    deshadowed_img = cv2.merge((img_b, img_g, img_r))

    # calculate rmse
    rmse = np.sqrt(np.mean((original_img - deshadowed_img)**2))

    # calculate ssim
    ssim_val = ssim(original_img, deshadowed_img, multichannel=True, win_size=3, channel_axis=2)

    # calculate psnr
    psnr = 10 * np.log10(255**2 / rmse**2)

    # print metrics
    print(f'used time = {time.time() - t0}')
    print(f'rmse = {rmse}')
    print(f'ssim = {ssim_val}')
    print(f'psnr = {psnr}')

    # display
    img_display = cv2.hconcat((original_img, deshadowed_img))
    img_display = cv2.resize(img_display, (0, 0), fx=1/resize_factor, fy=1/resize_factor)

    cv2.imshow(f"original vs remove shadow", img_display)
    
    cv2.imwrite(f"datasets/result/{image_name}", img_display)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
