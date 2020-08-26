from Pokedex import Pokédex
from dcnet import Discriminator, Generator

import torch
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
parser.add_argument('--gnet', type=str, default='checkpoints/g/Pokemonet_sg.pth400',
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--dnet', type=str, default='checkpoints/d/Pokemonet_sd.pth400',
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--epochs', '-e', default=150, type=int,
                    help='the number of training epochs')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
args = parser.parse_args()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train_dnet(loader,gnet, dnet, optimizer):
    loss_amount = 0
    # load train data
    for iteration, batch in enumerate(loader):
        # forward & backprop
        optimizer.zero_grad()
        for p in dnet.parameters():
            p.data.clamp_(-0.02, 0.02)

        Pokémon = batch.permute(0,3,1,2).type(torch.float).cuda()
        size = Pokémon.size()[0]
        fakePoké = gnet(torch.randn(size, 100, 1, 1))

        preds_r = dnet(Pokémon).mean()
        preds_f = dnet(fakePoké).mean()
        
        alpha = np.random.rand() * 0.1 + 0.9
        beta = np.random.rand() * 0.1 + 0.9
        if np.random.rand() > 0.05:
            loss=-preds_r*alpha+preds_f*beta
        else:
            loss=preds_r*alpha-preds_f*beta
        loss.backward()
        optimizer.step()
        loss_amount += loss.item()
    print('Dnet Loss: %.6f' % (loss_amount/iteration))
    return loss_amount/iteration

def train_gnet(loader, gnet, dnet, optimizer):
    loss_amount = 0
    # load train data
    for iteration, batch in enumerate(loader):
        # forward & backprop
        optimizer.zero_grad()

        Pokémon = batch.permute(0,3,1,2).type(torch.float).cuda()
        size = Pokémon.size()[0]

        realPoké = gnet(torch.randn(size,100,1,1))
        
        loss = -dnet(realPoké).mean()
        loss.backward()

        optimizer.step()
        loss_amount += loss.item()
    print('Gnet Loss: %.6f' % (loss_amount/iteration))
    return loss_amount/iteration

def train():

    torch.backends.cudnn.benchmark = True
    Gnet = Generator().cuda()
    Dnet = Discriminator().cuda()

    optimizerD = optim.Adam(Dnet.parameters(), lr=args.lr*2)
    optimizerG = optim.Adam(Gnet.parameters(),lr=args.lr)
    criterion = nn.BCELoss().cuda()

    print('Loading the dataset...')
    data = Pokédex()
    PokéBall = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print('Training GhostNet on:', data.name)
    print('Using the specified args:')
    print(args)

    for iteration in range(args.start_iter + 1, args.epochs + 1):
        print('Epoch %d' % iteration)
        Dnet.train()
        Gnet.eval()
        loss_d = train_dnet(PokéBall, Gnet, Dnet, optimizerD)
        Dnet.eval()
        Gnet.train()
        loss_g = train_gnet(PokéBall, Gnet, Dnet, optimizerG)
        data.shuffle()
        if not (iteration-args.start_iter) == 0 and iteration % 50 == 0:
            torch.save(Dnet.state_dict(),
                        ops('checkpoints', 'd', 'Pokemonet_sd.pth%03d' % iteration))
            torch.save(Gnet.state_dict(),
                        ops('checkpoints', 'g', 'Pokemonet_sg.pth%03d' % iteration))
if __name__ == '__main__':
    train()