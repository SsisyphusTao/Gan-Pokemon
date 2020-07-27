import torch
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
from os import listdir

mean = np.array([79.3103, 82.8892, 88.3397], np.float32)
std = np.array([54.4788, 57.9199, 61.1742], np.float32)

class Pokédex(Dataset):
    def __init__(self):
        super().__init__()
        self.Pokémons = []
        for i in listdir('Pokemons'):
            self.Pokémons.append((cv.resize(cv.imread('Pokemons/'+i), (128,128)).astype(np.float32) - mean) / std)

    def __len__(self):
        return len(self.Pokémons)
    
    def __getitem__(self, index):
        pokémon = self.Pokémons[index]
        a = np.random.randint(90)
        m = cv.getRotationMatrix2D((64,64), a, 1)
        pokémon = cv.warpAffine(pokémon, m, (128,128))

        if np.random.randint(4):
            return cv.flip(pokémon, np.random.randint(2)-1)
        else:
            return pokémon

    name = 'Pokedex'