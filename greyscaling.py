import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

src_path = "C:/Users/PheoNex/Documents/python programs/ocr_img/"


def rgb_to_gray(img):
    grayImage = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    Avg = (R + G + B)
    grayImage = img.copy()

    for i in range(3):
        grayImage[:, :, i] = Avg

    return grayImage


def greytobinary(img):
    # (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 130
    ret,im_bw = cv2.threshold(img, thresh, 256, cv2.THRESH_BINARY)
    cv2.imwrite(src_path + 'binary_image.png', im_bw)
    return im_bw


# def surround(x, y,h,w):
#       for x in range(0, h):
#              for y in range(0, w):
#                     if x-1

def invert_img(Img):
    invert = np.zeros(Img.shape)
    h = Img.shape[0] - 1
    w = Img.shape[1] - 1
    # invert = greyImg.copy()
    c = 0

    for x in range(0, h):
        for y in range(0, w):
            if x > 0 and y > 0 and x + 1 <= h and y + 1 <= w:
                if Img[x - 1, y - 1] == 0:
                    c = c + 1
                if Img[x - 1, y] == 0:
                    c = c + 1
                if Img[x - 1, y + 1] == 0:
                    c = c + 1
                if Img[x, y - 1] == 0:
                    c = c + 1
                if Img[x, y + 1] == 0:
                    c = c + 1
                if Img[x + 1, y - 1] == 0:
                    c = c + 1
                if Img[x + 1, y] == 0:
                    c = c + 1
                if Img[x + 1, y + 1] == 0:
                    c = c + 1
            if c > 4:
                invert[x, y] = 255
                c = 0
            else:
                invert[x, y] = 0
                c = 0

    cv2.imwrite(src_path + 'invert_image.png', invert)
    return invert


image = cv2.imread('C:/Users/PheoNex/Documents/python programs/ocr_img/colour.jpg')
grayImage = rgb_to_gray(image)
cv2.imwrite(src_path + "11_grey.png", grayImage)
plt.imshow(grayImage)
plt.show()
greyImg = cv2.imread(src_path + '2.png', cv2.IMREAD_GRAYSCALE)
binaryImg = greytobinary(greyImg)
plt.imshow(binaryImg)
plt.show()
binImg = cv2.imread(src_path + '2.png', cv2.IMREAD_UNCHANGED)
invertImg = invert_img(binaryImg)
plt.imshow(invertImg)
plt.show()
