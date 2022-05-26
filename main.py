# Autores: Gabriel Baumgarten e Henrique Da Gama
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
from turtle import width
from cv2 import ADAPTIVE_THRESH_GAUSSIAN_C, dilate
import numpy as np
import cv2
from matplotlib import pyplot as plt
sys.setrecursionlimit(3000)

INPUT_IMAGE =  './img/8.bmp'
ALTURA_MIN = 5
LARGURA_MIN = 5
N_PIXELS_MIN = 5
THRESHOLD = 0.80


def calcula_verdice(img):    
    rows, cols, channels = img.shape    
    #grau_verdice = np.empty ((rows, cols, 1), np.float32)
    grau_verdice = np.zeros((rows, cols,3),dtype=np.float32)

    #Cria a mascara
    for i in range(rows):
        for j in range(cols):
            aux = max(img[i, j, 0], img[i, j, 2])
            if img[i, j, 1] > aux:
                grau_verdice[i, j] = 1 - (img[i, j, 1] - aux)
            else: 
                grau_verdice[i, j] = 1    
               
    cv2.imshow("Mascara",grau_verdice)

    #Normaliza a mascara
    #grau_verdice2 = np.empty ((rows, cols, 1), np.float32)
    grau_verdice2 = np.zeros((rows, cols,3),dtype=np.float32)
    #grau_verdice2 = np.zeros(grau_verdice.shape, dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            # grau_verdice2[i, j] = (grau_verdice[i, j] - grau_verdice[1][1])/(grau_verdice.max() - grau_verdice[1][1])
            grau_verdice2[i, j] = (grau_verdice[i, j] - THRESHOLD)/(grau_verdice.max() - THRESHOLD)

    cv2.imshow("MascaraNormalizada",grau_verdice2)

    grau_verdice2 = cv2.erode(grau_verdice2, (3,3))
    cv2.imshow("Erodido",grau_verdice2)

    grau_verdice2 = cv2.GaussianBlur(grau_verdice2, (3,3), 0)
    cv2.imshow("Blur",grau_verdice2)

    #Separa a tela verde da imagem
    teste = grau_verdice2 * img
    #teste = cv2.multiply(grau_verdice2, img)
    #img [grau_verdice2 == 0] = 255
    #img = cv2.bitwise_and(img, img, mask=grau_verdice2)
    cv2.imshow("FrenteChroma",teste)

    #"Libera" espaço no fundo
    fundo = cv2.imread ('./img/Capturar.png', cv2.IMREAD_COLOR)
    fundo = cv2.resize(fundo, (grau_verdice2.shape[1],grau_verdice2.shape[0]))
    fundo = fundo.astype(np.float32)/255
    fundo = fundo * (1 - grau_verdice2)
    cv2.imshow("FundoChroma",fundo)    

    #Junta fundo e frente
    chormaKey = fundo + teste
    cv2.imshow("Chroma",chormaKey)
    cv2.imwrite("b.png", chormaKey*255)

def main ():

    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img_float = img.astype(np.float32)/255 
    width = round(img_float.shape[1]/4)
    height = round(img_float.shape[1]/4)
    img_float = cv2.resize(img_float, (width, height))
    cv2.imshow('Original', img)    

    # grau_verdice = img_float[:,:,1].reshape((img_float.shape[0],img_float.shape[1],1))
    # mask_green = np.where(grau_verdice < 0.2, img_float, 0)
    mask_green = calcula_verdice(img_float)
    # cv2.imshow('grau_verdice', mask_green)    

    cv2.waitKey ()
    cv2.destroyAllWindows ()
    
if __name__ == '__main__':
    main ()