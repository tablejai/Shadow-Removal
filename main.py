import cv2
import numpy as np
from bg_prop import *
from separate import *


def main():
    # cv2.namedWindow("whatever", cv2.WINDOW_NORMAL)
    img = cv2.imread("datasets/003_017.jpg")
    # cv2.imshow("Dataset img", img)
    diff, shadow_mask = bg_prop(img)
    separate_umbra(img, shadow_mask)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
