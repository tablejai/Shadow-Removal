import numpy as np
import cv2


def separate_umbra(img, shadow_mask, iters=3, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, dtype=np.uint8)
    umbra = cv2.dilate(shadow_mask, kernel, iterations=iters)
    umbra = cv2.bitwise_not(umbra)
    penumbra_mask = cv2.bitwise_not(cv2.bitwise_or(umbra, shadow_mask))
    umbra_mask = cv2.bitwise_not(cv2.bitwise_or(shadow_mask, penumbra_mask))

    cv2.imshow("penumbra", penumbra_mask)
    cv2.imshow("umbra_mask", umbra_mask)
    # print("penumbra shape", penumbra_mask.shape)
    # print("img shape", img.shape)
    for i in range(penumbra_mask.shape[0]):
        for j in range(penumbra_mask.shape[1]):
            if penumbra_mask[i][j]:
                img[i][j][0] += 100

    for i in range(umbra_mask.shape[0]):
        for j in range(umbra_mask.shape[1]):
            if umbra_mask[i][j]:
                img[i][j][2] += 70
    img = np.asanyarray(img)

    cv2.imshow("rgb_mask", img)
