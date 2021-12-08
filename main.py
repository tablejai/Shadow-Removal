import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from plot_helper import plot_histogram_1Channel, plot_histogram_rgb


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
    diff = original_img.astype(np.int8) - background.astype(np.int8)
    return diff


def global_background_mean(img, mask):
    t = cv2.mean(img, mask)[0]
    cv2.imshow("img", img)
    cv2.imshow("mask", mask)
    print("gloabl mean", t)
    return t


def normalize_shodowed_text(img, mask, offset, title=""):
    result = img
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i, j]:
                result[i, j] += offset
                # if result[i, j] < 0:
                #     result[i, j] = 0

    return result


def normalize(orig_img, img, title=""):

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.xlim([0, 256])
    plt.title(title+"_original")

    y_hist, x_hist, _ = plt.hist(np.ravel(orig_img), density=True, bins=256)

    plt.subplot(1, 3, 2)
    plt.hist(np.ravel(img), density=True, bins=256)
    plt.title(title+"_subtracted")

    offset = x_hist[np.where(y_hist == y_hist.max())]
    img = img + offset

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < 0:
                img[i, j] = 0

    plt.subplot(1, 3, 3)
    plt.xlim([0, 256])
    plt.title(title+"_normalized")
    plt.hist(np.ravel(img), density=True, bins=256)

    return img


def min_max_filtering(original_img, size, title=""):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    shadow = cv2.morphologyEx(
        original_img, cv2.MORPH_CLOSE, kernel, iterations=4)
    cv2.imshow("shadow"+title, shadow)

    diff = background_subtraction(original_img, shadow, title)

    shadow_mask = 255-cv2.threshold(shadow.astype(np.uint8), 110, 255, cv2.THRESH_BINARY)[
        1]

    normed = normalize(original_img, diff, title)

    text_mask = 255-cv2.threshold(normed.astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[
        1]

    t1 = cv2.mean(original_img, ~shadow_mask)[0]
    # TODO: here we should use local shadow mean instead of global
    t2 = cv2.mean(original_img, shadow_mask)[0]
    offset = t1-t2

    normed2 = normalize_shodowed_text(
        normed, shadow_mask & text_mask, offset, title)

    return normed2


if __name__ == '__main__':
    img = cv2.imread('datasets/test7.jpg')
    img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)

    t0 = time.time()

    img_b, img_g, img_r = cv2.split(img)

    img_b = min_max_filtering(img_b, 5, "B").astype(np.uint8)
    img_g = min_max_filtering(img_g, 5, "G").astype(np.uint8)
    img_r = min_max_filtering(img_r, 5, "R").astype(np.uint8)
    # print(f'used time = {time.time() - t0}')

    merged = cv2.merge((img_b, img_g, img_r))
    final_img = cv2.hconcat((img, merged))
    cv2.imshow("final_img", final_img)
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
