import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import json
import os

src_path = "C:/Users/PheoNex/Documents/python programs/ocr_img/"
data = {}
i = 0;

def walsh_transfer(im_mat , wal_mat):
    q = 0
    r = 0
    list_val = []
    while q < 64:
        while r < 64:
            imat = im_mat[q:q + 8, r:r + 8]

            Wmat = wal_mat[q:q + 8, r:r + 8]
            list_val.append(int(np.sum(np.array(imat) * np.array(Wmat))))
            r = r + 8
        q = q + 8
        r = 0
    return list_val


def Hadamard2Walsh(n):
    # Function computes both Hadamard and Walsh Matrices of n=2^M order
    # (c) 2015 QuantAtRisk.com, coded by Pawel Lachowicz, adopted after
    # au.mathworks.com/help/signal/examples/discrete-walsh-hadamard-transform.html
    #from scipy.linalg import hadamard
    from math import log

    hadamardMatrix = hadamard(n)
    HadIdx = np.arange(n)
    M = int(log(n, 2)) + 1

    for i in HadIdx:
        s = format(i, '#032b')
        s = s[::-1];
        s = s[:-2];
        s = list(s)
        x = [int(x) for x in s]
        x = np.array(x)
        if (i == 0):
            binHadIdx = x
        else:
            binHadIdx = np.vstack((binHadIdx, x))

    binSeqIdx = np.zeros((n, M)).T

    for k in reversed(range(1, int(M))):
        tmp = np.bitwise_xor(binHadIdx.T[k], binHadIdx.T[k - 1])
        binSeqIdx[k] = tmp

    tmp = np.power(2, np.arange(M)[::-1])
    tmp = tmp.T
    SeqIdx = np.dot(binSeqIdx.T, tmp)

    j = 1
    for i in SeqIdx:

        if (j == 1):
            walshMatrix = hadamardMatrix[int(i)]
        else:
            walshMatrix = np.vstack((walshMatrix, hadamardMatrix[int(i)]))
        j += 1

    return hadamardMatrix, walshMatrix

def greytobinary(img):
    # (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 130
    im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(src_path + 'binary_image.png', im_bw)
    plt.imshow(im_bw)
    plt.show()
    img_copy = np.copy(img)
    img_copy[img_copy[:] <= 127] = 0
    img_copy[img_copy[:] > 127] = 1
    thresh1 = np.copy(img_copy)
    height, width = img_copy.shape
    for i in range(height):
        if i - 1 >= 0 and i + 1 < height:
            for j in range(width):
                if j - 1 >= 0 and j + 1 < width:
                    if thresh1[i, j] == 0:  # logic for black
                        if thresh1[i - 1, j - 1] == 1 and thresh1[i, j - 1] == 1 and thresh1[i + 1, j - 1] == 1 and \
                                thresh1[
                                    i - 1, j + 1] == 1 and thresh1[i, j + 1] == 1 and thresh1[i + 1, j + 1] == 1:
                            thresh1[i - 1, j - 1] = 0
                            thresh1[i, j - 1] = 0
                            thresh1[i + 1, j - 1] = 0
                            thresh1[i - 1, j] = 0
                            thresh1[i + 1, j] = 0
                            thresh1[i - 1, j + 1] = 0
                    if thresh1[i, j] == 1:  # logic for white
                        if thresh1[i - 1, j - 1] == 0 and thresh1[i, j - 1] == 0 and thresh1[i + 1, j - 1] == 0 and \
                                thresh1[
                                    i - 1, j + 1] == 0 and thresh1[i, j + 1] == 0 and thresh1[i + 1, j + 1] == 0:
                            thresh1[i - 1, j - 1] = 1
                            thresh1[i, j - 1] = 1
                            thresh1[i + 1, j - 1] = 1
                            thresh1[i - 1, j] = 1
                            thresh1[i + 1, j] = 1
                            thresh1[i - 1, j + 1] = 1

    plt.imshow(thresh1)
    plt.show()
    return thresh1

def hline_image(img,wal):
    global  i
    start = 0
    endline = 0
    h = img.shape[0]
    w = img.shape[1]
    l = 0
    c = 0
    for m in range(w):
        for n in range(h):
            # if x > 0 and y > 0 and x+1 <= h and y+1 <= w :
            if img[n, m] != 1 and start == 0:
                start = m
                break
            if img[n, m] != 1 and start != 0:
                c = m;
            if img[n, m] == 1 and start != 0 and m != c:
                if n == h - 1:
                    l = m;
        if l != 0:
            endline = m - 1
            cropped_image = img[0:h - 1, start:endline]
            # cropped_image = img[0:h, :m + w]

            if start != endline:
                cv2.imwrite(src_path + 'capcrop.png', cropped_image)
                plt.imshow(cropped_image)
                plt.show()
                i = i + 1
                dim = (64, 64)
                resized = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)
                walsh_trans = walsh_transfer(np.array(resized), wal)
                #plt.imshow(walsh_trans)
                #plt.show()
                #data[i] = []
                data[i] = np.array(walsh_trans).tolist()
                filePathNameWExt = src_path + 'datafile.json'
                with open(filePathNameWExt , 'w') as outfile:
                    # data = json.load(filePathNameWExt)
                    if os.stat(filePathNameWExt).st_size != 0:
                        data.update(data[i])
                    # filePathNameWExt.seek(0)
                    # json.dump(data, filePathNameWExt)
                    json.dump(data, outfile)
                #plt.imshow(walsh_trans)
                #plt.show()
            start = 0
            l = 0

def vline_image(Img, wal):
    #copyimg = Img.shape;
    start = 0
    endline = 0
    h = Img.shape[0]
    w = Img.shape[1]

    # line = np.zeros(shape = (start, endline + 1), dtype = 'int')
    # for x in range(0, 35):
    #       for y in range(0, w):
    #              line[x,y] = 255

    l = 0
    n = 0
    # invert = greyImg.copy()
    c = 0

    for x in range(0, h):
        for y in range(0, w):
            # if x > 0 and y > 0 and x+1 <= h and y+1 <= w :
            if Img[x, y] != 1 and start == 0:
                start = x
                break
            if Img[x, y] != 1 and start != 0:
                c = x;
            if Img[x, y] == 1 and start != 0 and x != c:
                if y == w-1:
                    l = x;
        if l > 0:
            endline = x
            cropped_image = Img[start:endline, 0:w]
            cv2.imwrite(src_path + 'capcrop.png', cropped_image)
            plt.imshow(cropped_image)
            plt.show()
            hline = cv2.imread(src_path + "capcrop.png", cv2.IMREAD_UNCHANGED)
            hlineimg = hline_image(hline,wal)
            start = 0
            l = 0

   # return cropped_image

n = 64
(H,W)=Hadamard2Walsh(n)

vline = cv2.imread(src_path + "trimg.jpg", cv2.IMREAD_GRAYSCALE)
binimg = greytobinary(vline)
vlineimg = vline_image(binimg, W)

#plt.imshow(hlineimg)
#plt.show()