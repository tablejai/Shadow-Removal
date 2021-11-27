import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def maxkernel(img, size):
    size = size//2
    result = cv.copyMakeBorder(img, size, size, size, size, cv.BORDER_REPLICATE)
    for i in range(size, img.shape[0]-size):
        for j in range(size, img.shape[1]-size):
            result[i, j] = np.max(img[i-size:i+size+1, j-size:j+size+1])
    return result[size:-size, size:-size]


def minkernel(img, size):
    size = size//2
    result = cv.copyMakeBorder(img, size, size, size, size, cv.BORDER_REPLICATE)
    for i in range(size, img.shape[0]-size):
        for j in range(size, img.shape[1]-size):
            result[i, j] = np.min(img[i-size:i+size+1, j-size:j+size+1])
    return result[size:-size, size:-size]

def shadowDetect(V_max, V_min):
    alpha = ((V_max-V_min).astype(np.float32)/(V_max).astype(np.float32))
    L = (V_max*(1-alpha)).astype(np.float32) + (V_min*alpha).astype(np.float32)
    return L

target = "001_001"
img = cv.imread(target+".jpg", cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (0,0), fx = 1, fy = 1)
print("image shape:", img.shape)


L = img
L3 = img
for i in range(1):
    if i == 3:
        L3  = shadowDetect(V_max, V_min)
    V_max = maxkernel(L, 3)
    V_min = minkernel(L, 3)
    L = shadowDetect(V_max, V_min)
    
    
cv.imshow("origin", img)

ret, threshed = cv.threshold(L, 127, 255, cv.THRESH_BINARY)
dialation_kernel = np.ones((3,3), np.uint8)
dialated = cv.dilate(threshed, dialation_kernel, iterations = 1)
penumbra_mask = abs(threshed - dialated)
umbra_mask = abs(255-(threshed-penumbra_mask))
unshadowed_mask = abs(255 - (penumbra_mask + umbra_mask))
shadowed_mask = 255 - unshadowed_mask
cv.imshow("Shadow", L.astype(np.uint8))
cv.imshow("umbra", umbra_mask)
cv.imshow("pernubra", penumbra_mask)
cv.imshow("unshadowed", unshadowed_mask)
cv.imshow("shadowed", shadowed_mask)

G = cv.mean(img[unshadowed_mask == 255]);
G_mat = np.full(L.shape, G[0])
r = G_mat/L


recon = img
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if shadowed_mask[i][j] == 255:
            if(r[i][j] < 2 and img[i][j] < 300):
                recon[i][j] = r[i][j]*img[i][j] 
            else:
                continue
        else:
            recon[i][j] = img[i,j]   
cv.imshow("recon", recon)
cv.waitKey(0)
cv.destroyAllWindows()