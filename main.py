# Autores: Gabriel Baumgarten e Henrique Da Gama
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
from cv2 import ADAPTIVE_THRESH_GAUSSIAN_C, dilate
import numpy as np
import cv2
from matplotlib import pyplot as plt
sys.setrecursionlimit(3000)

INPUT_IMAGE =  './img/0.bmp'
ALTURA_MIN = 5
LARGURA_MIN = 5
N_PIXELS_MIN = 5
THRESHOLD = 0.7


def calcula_verdice(img):
    
    rows, cols, channels = img.shape    
    grau_verdice = np.empty ((rows, cols, 1), np.float32)
    for row in range(rows):
        for column in range(cols):
            grau_verdice[row][column] = img[row][column][1]*0.5870/sum(img[row][column])
            # grau_verdice[row][column] = round(img[row][column][1]*0.5870/sum([img[row][column][0]*0.2989, img[row][column][1]*0.5870, img[row][column][2]*0.1140]), 2)

    return grau_verdice

def main ():

    # images = ['60.bmp','82.bmp','114.bmp','150.bmp','205.bmp']

    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img_float = img.astype(np.float32)/255 
    cv2.imshow('Original', img)    

    # grau_verdice = img_float[:,:,1].reshape((img_float.shape[0],img_float.shape[1],1))
    # mask_green = np.where(grau_verdice < 0.2, img_float, 0)
    mask_green = calcula_verdice(img_float)
    # aux = np.where(mask_green < 0.7, img_float, 0)
    # mask_green = 1 - mask_green
    aux = mask_green*255
    # rows, cols, channels = img.shape
    # for row in range(rows):
    #     for column in range(cols):
    #         aux[row][column][0]  = round((1-mask_green[row][column][0])*255)
    cv2.imshow('grau_verdice', aux)    

    # grau_verdice = np.empty ((rows, cols, 1), np.float32)
    # img=cv2.normalize(img, normalizedImg, 0, 0.7, cv2.NORM_MINMAX)
    # cv2.imshow('Normalizada', img)    

   

    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()