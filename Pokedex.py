import torch
from torch.utils.data import Dataset
import cv2 as cv
from os import listdir

class Pokédex(Dataset):
    def __init__(self):
        super().__init__()
        self.Pokémons = []
        for i in listdir('Pokémons'):
            self.Pokémons.append(cv.imread('Pokémons/'+i))

    def __len__(self):
        return len(self.Pokémons)
    
    def __getitem__(self, index):
        return self.Pokémons[index]

    name = 'Pokédex'