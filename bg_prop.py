import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2
import copy


def pool_2d(img: np.ndarray, kernel_size=3, stride=1, padding=0, pool="max"):
    img = cv2.copyMakeBorder(
        img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    tar_shape = ((img.shape[0] - kernel_size) // stride + 1,
                 (img.shape[1] - kernel_size) // stride + 1)

    shape_w = (tar_shape[0], tar_shape[1], kernel_size, kernel_size)
    strides_w = (stride*img.strides[0], stride *
                 img.strides[1], img.strides[0], img.strides[1])

    img_w = as_strided(img, shape_w, strides_w)
    # if pool == 'max':
    #     return img_w.max(axis=3)
    # elif pool == 'avg':
    #     return img_w.mean(axis=3)
    # elif pool == 'min':
    #     return img_w.min(axis=3)

    if pool == 'max':
        return img_w.max(axis=(2, 3))
    elif pool == 'avg':
        return img_w.mean(axis=(2, 3))
    elif pool == 'min':
        return img_w.min(axis=(2, 3))


def fusion_factor(max_kernel, min_kernel):
    return (max_kernel - min_kernel) / max_kernel


def est_bg(img, iter):
    output_img = copy.deepcopy(img)
    for _ in range(iter):
        max_kernel = pool_2d(output_img, kernel_size=5, pool="max")
        cv2.imshow("max", max_kernel)
        min_kernel = pool_2d(output_img, kernel_size=5, pool="min")
        cv2.imshow("min", min_kernel)

        alpha = fusion_factor(max_kernel, min_kernel)
        reverse_alpha = np.subtract(np.ones(alpha.shape), alpha)
        print("reverse_alpha", reverse_alpha)

        alpha_max = np.multiply(max_kernel, reverse_alpha,
                                out=max_kernel, casting="unsafe")
        print("alpha max", alpha_max)
        cv2.imshow("alpha max", alpha_max)

        alpha_min = np.multiply(
            min_kernel, alpha, out=min_kernel, casting="unsafe")
        print("alpha min", alpha_min)
        cv2.imshow("alpha min", alpha_min)

        output_img = alpha_max + alpha_min
        cv2.imshow("output img", output_img)
    return output_img


def shadow_binarization(img: np.ndarray):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, filtered = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return filtered


def remove_small_dots(img):
    output = img.copy()
    # Clear small blck dots with dilation
    kernel = np.ones((3, 3), dtype=np.uint8)
    res = cv2.dilate(output, kernel, iterations=1)
    return res


def fill_holes(img):
    output = img.copy()
    des = cv2.bitwise_not(output)
    contours, hierachy = cv2.findContours(
        des, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contours first
    des = cv2.drawContours(des, contours, -1, 255, -1)
    gray = cv2.bitwise_not(des)

    # Use opening with a 13 * 13 elipse to find the shadow
    # Maybe can try tune the size IDK
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    res = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    return res


def bg_prop(img: np.ndarray):
    temp = copy.deepcopy(img)

    # I thought you do it like this
    # temp = est_bg(temp, 3)

    # Turns out this looks like it makes more sense
    for _ in range(3):
        temp = pool_2d(temp, kernel_size=3, pool="max")

    # cv2.imshow("est", temp)

    # temp = cv2.equalizeHist(temp)

    temp = shadow_binarization(temp)
    temp = remove_small_dots(temp)
    temp = fill_holes(temp)
    # cv2.imshow("bin", shadow_binarization(temp))
    # cv2.imshow("hole filled", temp)

    diff = cv2.bitwise_and(img, cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR))
    cv2.imshow("diff", diff)
    return diff, temp
    # return max_kernel
    # return temp
