from Pokedex import Pokédex
from dcnet import Discriminator, Generator

import torch
import torchvision
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
from os.path import join as ops

parser = argparse.ArgumentParser(
    description="Pokémon, Getto Daze!")
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size', default=128, type=int,
                    help='Batch size for training')
parser.add_argument('--gnet', type=str, default=None,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--dnet', type=str, default=None,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--epochs', '-e', default=8000, type=int,
                    help='the number of training epochs')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    help='initial learning rate')
args = parser.parse_args()
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train_dnet(loader,gnet, dnet, optimizer):
    loss_amount = 0
    # load train data
    for iteration, batch in enumerate(loader):
        # forward & backprop
        optimizer.zero_grad()
        for p in dnet.parameters():
            p.data.clamp_(-0.02, 0.02)

        Pokémon = batch[0].cuda()
        fakePoké = gnet(torch.randn(args.batch_size, 100, 1, 1).cuda())

        preds_r = dnet(Pokémon)
        preds_f = dnet(fakePoké)
        
        loss = -preds_r+preds_f
        loss.backward()
        optimizer.step()
        loss_amount += loss.item()
    print('Dnet Loss: %.6f' % (loss_amount/iteration))
    return loss_amount/iteration

def train_gnet(loader, gnet, dnet, optimizer):
    loss_amount = 0
    # load train data
    for iteration, _ in enumerate(loader):
        # forward & backprop
        optimizer.zero_grad()
        realPoké = gnet(torch.randn(args.batch_size,100,1,1).cuda())
        
        loss = -dnet(realPoké)
        loss.backward()

        optimizer.step()
        loss_amount += loss.item()
    print('\033[31mGnet Loss:\033[32m %.6f\033[0m' % (loss_amount/iteration))
    return loss_amount/iteration

def train():

    torch.backends.cudnn.benchmark = True
    Gnet = Generator().cuda()
    Dnet = Discriminator().cuda()

    optimizerD = optim.RMSprop(Dnet.parameters(), lr=args.lr*4)
    optimizerG = optim.RMSprop(Gnet.parameters(), lr=args.lr)

    print('Loading the dataset...', end='')
    Pokedex = torchvision.datasets.ImageFolder(root='/ai/ailab/User/huangtao/Gan-Pokemon/imgs',
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Resize(64),
                                    torchvision.transforms.CenterCrop(64),
                                    torchvision.transforms.RandomRotation(5),
                                    torchvision.transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    PokéBall = DataLoader(Pokedex, batch_size=args.batch_size, shuffle=True, num_workers=16)
    print('Done!')
    print('Training GAN on:', 'Pokedex')
    print('Using the specified args:')
    print(args)

    for i in range(20):
        loss_d = train_dnet(PokéBall, Gnet, Dnet, optimizerD)

    for iteration in range(args.start_iter + 1, args.epochs + 1):
        print('Epoch %d' % iteration)
        for _ in range(5):
            loss_d = train_dnet(PokéBall, Gnet, Dnet, optimizerD)
        loss_g = train_gnet(PokéBall, Gnet, Dnet, optimizerG)
        if not (iteration-args.start_iter) == 0 and iteration % 100 == 0:
            torch.save(Dnet.state_dict(),
                        ops('checkpoints', 'd', 'Pokemonet_sd.pth%04d' % iteration))
            torch.save(Gnet.state_dict(),
                        ops('checkpoints', 'g', 'Pokemonet_sg.pth%04d' % iteration))
if __name__ == '__main__':
    train()