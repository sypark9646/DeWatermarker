import numpy as np
import cv2
import os
#this code is from https://stackoverflow.com/questions/23238308/how-to-remove-single-pixels-on-the-borders-of-a-blob/23239190

def hitmiss(src, kernel):
    im = src / 255
    k1 = (kernel == 1).astype('uint8')
    k2 = (kernel == -1).astype('uint8')
    e1 = cv2.erode(im, k1, borderType=cv2.BORDER_CONSTANT)
    e2 = cv2.erode(1-im, k2, borderType=cv2.BORDER_CONSTANT)
    e = np.logical_and(np.array(e1), np.array(e2))
    e[e == True] = 1
    e[e == False] = 0
    return e

path_of_images = './image1/image1_corner_connectedComponent/1/'
path_of_outputimages = './image1/image1_corner_connectedComponent/1_corner'
list_of_images = os.listdir(path_of_images)

for image in list_of_images:
    im_binary = cv2.imdecode(np.fromfile(os.path.join(path_of_images, image), dtype=np.uint8) , cv2.IMREAD_GRAYSCALE) #한글파일명

    kernel = np.array([[-1,-1, 1], 
                       [-1, 1, 1], 
                       [-1,-1, 1]])

    im_mask = np.zeros(im_binary.shape, np.uint8)

    im_mask |= hitmiss(im_binary, kernel)
    im_mask |= hitmiss(im_binary, np.fliplr(kernel))
    im_mask |= hitmiss(im_binary, kernel.T)
    im_mask |= hitmiss(im_binary, np.flipud(kernel.T))

    im_dst = im_binary & ((1 - im_mask) * 255)

    is_success, im_buf_arr = cv2.imencode(".tiff", im_dst)
    im_buf_arr.tofile(os.path.join(path_of_outputimages, image))
