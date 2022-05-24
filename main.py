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

INPUT_IMAGE =  './img/2.bmp'
ALTURA_MIN = 5
LARGURA_MIN = 5
N_PIXELS_MIN = 5
THRESHOLD = 0.95


def calcula_verdice(img):    
    rows, cols, channels = img.shape    
    grau_verdice = np.empty ((rows, cols, 1), np.float32)

    #Criar mascara
    for i in range(rows):
        for j in range(cols):
            aux = max(img[i, j, 0], img[i, j, 2])
            if img[i, j, 1] > aux:
                grau_verdice[i, j] = 1 - (img[i, j, 1] - aux)
            else: 
                grau_verdice[i, j] = 1

    #Remover fundo verde
    # for i in range(rows):
    #     for j in range(cols):
    #         if grau_verdice[i, j] < THRESHOLD:
    #             img[i, j] = grau_verdice[i, j]

    
               
    cv2.imshow("imagem",grau_verdice)
    # cv2.imshow("imagemm",img)


    grau_verdice2 = np.empty ((rows, cols, 1), np.float32)
    for i in range(rows):
        for j in range(cols):
            grau_verdice2[i, j] = (grau_verdice[i, j] - grau_verdice[1][1])/(grau_verdice.max() - grau_verdice[1][1])

    cv2.imshow("imagem_normalizada",grau_verdice2)


    teste = grau_verdice2 * img
    cv2.imshow("testeFrenteChroma",teste)

    fundo = cv2.imread ('./img/Capturar.png', cv2.IMREAD_COLOR)
    fundo = cv2.resize(fundo, (grau_verdice2.shape[1],grau_verdice2.shape[0]))
    fundo = fundo.astype(np.float32)/255
    fundo = fundo * (1 - grau_verdice2)
    cv2.imshow("testeFundoChroma",fundo)

    chormaKey = fundo + teste
    cv2.imshow("testeChroma",chormaKey)


def main ():

    # images = ['60.bmp','82.bmp','114.bmp','150.bmp','205.bmp']

    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img_float = img.astype(np.float32)/255 
    cv2.imshow('Original', img)    

    # grau_verdice = img_float[:,:,1].reshape((img_float.shape[0],img_float.shape[1],1))
    # mask_green = np.where(grau_verdice < 0.2, img_float, 0)
    mask_green = calcula_verdice(img_float)
    cv2.imshow('grau_verdice', mask_green)    


    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()