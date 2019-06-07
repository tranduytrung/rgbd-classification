import numpy as np
import numba

def crop(np_arr, y, x, h, w):
    return np.copy(np_arr[y:y + h, x:x + w])


@numba.njit(numba.float32[:](numba.float32[:, :, :], numba.float32, numba.float32))
def get_bilinear_pixel(imArr, posX, posY):
    out = np.empty(imArr.shape[2], dtype=np.float32)

    # Get integer and fractional parts of numbers
    modXi = int(posX)
    modYi = int(posY)
    modXf = posX - modXi
    modYf = posY - modYi
    modXiPlusOneLim = min(modXi+1, imArr.shape[1]-1)
    modYiPlusOneLim = min(modYi+1, imArr.shape[0]-1)

    # Get pixels in four corners
    for chan in range(imArr.shape[2]):
        bl = imArr[modYi, modXi, chan]
        br = imArr[modYi, modXiPlusOneLim, chan]
        tl = imArr[modYiPlusOneLim, modXi, chan]
        tr = imArr[modYiPlusOneLim, modXiPlusOneLim, chan]

        # Calculate interpolation
        b = modXf * br + (1. - modXf) * bl
        t = modXf * tr + (1. - modXf) * tl
        pxf = modYf * t + (1. - modYf) * b
        out[chan] = pxf

    return out

@numba.njit(numba.float32[:, :, :](numba.float32[:, :, :], numba.typeof((0, 0))))
def resize(arr, size):
    enlargedImg = np.empty((size[0], size[1], arr.shape[2]), dtype=np.float32)
    rowScale = float(arr.shape[0] - 1) / float(enlargedImg.shape[0] - 1)
    colScale = float(arr.shape[1] - 1) / float(enlargedImg.shape[1] - 1)

    for r in range(enlargedImg.shape[0]):
            orir = r * rowScale  # Find position in original image
            for c in range(enlargedImg.shape[1]):    
                oric = c * colScale
                enlargedImg[r, c] = get_bilinear_pixel(arr, oric, orir)
                
    return enlargedImg


if __name__ == "__main__":
    arr = np.array([[[1],[2]],[[3],[4]]], dtype=np.float32)
    resize(arr, (4,4))
