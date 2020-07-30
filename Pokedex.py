import torch
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
from os import listdir

mean = np.array([79.0846, 83.2913, 88.9984], np.float32)
std = np.array([52.4808, 57.0215, 59.5216], np.float32)

class Pokédex(Dataset):
    def __init__(self):
        super().__init__()
        self.Pokémons = []
        for i in listdir('Pokemons'):
            self.Pokémons.append((cv.imread('Pokemons/'+i).astype(np.float32) - mean) / std)
        self.shuffle()

    def __len__(self):
        return len(self.Pokémons)

    def __getitem__(self, index):
        pokémon = self.Pokémons[index]
        if np.random.randint(2):
            pokémon = pokémon[:,:,np.random.permutation(3)]
        a = np.random.randint(30) - 15 
        m = cv.getRotationMatrix2D((64,64), a, 1)
        pokémon = cv.warpAffine(pokémon, m, (128,128))
        if np.random.randint(2):
            return cv.flip(pokémon, np.random.randint(2)-1)
        else:
            return pokémon

    def shuffle(self):
        np.random.shuffle(self.Pokémons)
    name = 'Pokedex'