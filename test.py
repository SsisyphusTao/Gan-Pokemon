from Pokedex import Pokédex
from dcnet import Discriminator, Generator

import cv2 as cv
import numpy as np
import torch
from sys import argv
from os import listdir

mean = np.array([79.0846, 83.2913, 88.9984], np.float32)
std = np.array([52.4808, 57.0215, 59.5216], np.float32)
gnet = Generator()

gnet.load_state_dict({k.replace('module.',''):v 
            for k,v in torch.load('/Users/videopls/Desktop/Gan-Pokemon/Pokemonet_sg.pth'+argv[1], map_location=torch.device('cpu')).items()})
gnet.eval()
if len(argv)<3:
    with torch.no_grad():
        for i in listdir('/Users/videopls/Desktop/Gan-Pokemon/Pokemons'):
            # print(i)
            Pokemon = cv.imread('/Users/videopls/Desktop/Gan-Pokemon/Pokemons/'+i)#.astype(np.float32) - mean) / std
            # a = np.random.randint(90)
            # m = cv.getRotationMatrix2D((64,64), a, 1)
            # Pokemon = cv.warpAffine(Pokemon, m, (128,128))
            # core = dnet(torch.from_numpy(Pokemon).permute(2,0,1).unsqueeze(0))
            core = gnet(torch.randn(1,100, 1, 1))
            # img = gnet(core)
            img = core[0].permute(1,2,0).numpy() * 127.5 + 127.5
            img = cv.resize(img, (128,128))
            cv.imshow('f', img.astype(np.uint8))

            if cv.waitKey() == 27:
                break
else:
    n = int(argv[2])
    displayP = []
    displayF = []
    tempP = []
    tempF = []
    with torch.no_grad():
        for i in listdir('/Users/videopls/Desktop/Gan-Pokemon/Pokemons'):
            Pokemon = cv.imread('/Users/videopls/Desktop/Gan-Pokemon/Pokemons/'+i)
            tempP.append(Pokemon)
            if len(tempP)== n:
                displayP.append(tuple(tempP))
                tempP.clear()
            core = gnet(torch.randn(1,100, 1, 1))
            img = core[0].permute(1,2,0).numpy() * 127.5 + 127.5
            img = img.astype(np.uint8)
            tempF.append(img)
            if len(tempF) == n:
                displayF.append(tuple(tempF))
                tempF.clear()
            if len(displayF)==n:
                break
        for i in range(n):
            displayP[i] = cv.hconcat(displayP[i])
            displayF[i] = cv.hconcat(displayF[i])
        displayP = cv.vconcat(displayP)
        displayF = cv.vconcat(displayF)

        cv.imwrite('Pokemons.jpg', displayP)
        cv.imwrite('%s.jpg'%argv[1], displayF)

