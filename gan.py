from Pokedex import Pokédex
from resnet import getDiscriminator, getGenerator, snGenerator, snDiscriminator

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import time
from os.path import join as ops

parser = argparse.ArgumentParser(
    description="Pokémon, Getto Daze!")
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size for training')
parser.add_argument('--gnet', type=str, default='checkpoints/Pokemonet_sg.pth400',
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--dnet', type=str, default='checkpoints/Pokemonet_sd.pth400',
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--epochs', '-e', default=400, type=int,
                    help='the number of training epochs')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
args = parser.parse_args()
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def train_dnet(loader, dnet, sgnet, sdnet, optimizer):
    loss_amount = 0
    # load train data
    for iteration, batch in enumerate(loader):
        # forward & backprop
        optimizer.zero_grad()
        for p in sdnet.parameters():
            p.data.clamp_(-0.01, 0.01)
        
        Pokémon = batch.permute(0,3,1,2).type(torch.float).cuda()
        size = Pokémon.size()[0]

        Pokécore = dnet(Pokémon)
        fakePoké = sgnet(torch.randn(size, 100))

        preds_r = sdnet(Pokécore).mean()
        preds_f = sdnet(fakePoké).mean()

        loss = - preds_r * 0.5 + preds_f
        loss.backward()
        optimizer.step()
        loss_amount += loss.item()
    print('Dnet Loss: %.6f ' % (loss_amount/iteration))
    return loss_amount/iteration

def train_gnet(loader, sgnet, sdnet, optimizer):
    loss_amount = 0
    # load train data
    for iteration, batch in enumerate(loader):
        # forward & backprop
        optimizer.zero_grad()

        Pokémon = batch.permute(0,3,1,2).type(torch.float).cuda()
        size = Pokémon.size()[0]

        realPoké = sgnet(torch.randn(size*2,100))
        
        loss = -sdnet(realPoké).mean() * 0.75
        loss.backward()

        optimizer.step()
        loss_amount += loss.item()
    print('   Gnet Loss: %.6f' % (loss_amount/iteration))
    return loss_amount/iteration

def train_encoder_decoder(loader, dnet, gnet, criterion, optimizerD, optimizerG):
    loss_amount = 0
    # load train data
    for iteration, batch in enumerate(loader):
        # forward & backprop
        optimizerD.zero_grad()
        optimizerG.zero_grad()
        
        Pokémon = batch.permute(0,3,1,2).type(torch.float).cuda()
        cores = dnet(Pokémon)
        fakePoké = gnet(cores)

        loss = criterion(fakePoké, Pokémon)
        loss.backward()
        optimizerD.step()
        optimizerG.step()
        loss_amount += loss.item()
    print('Loss: %.6f ' % (loss_amount/iteration))
    return loss_amount/iteration

def train():

    torch.backends.cudnn.benchmark = True
    Gnet = getGenerator().cuda()
    Dnet = getDiscriminator(18).cuda()
    # Gnet.load_state_dict(torch.load('checkpoints/Pokemonet_g.pth'))
    # Dnet.load_state_dict(torch.load('checkpoints/Pokemonet_d.pth'))

    # sgnet = snGenerator(100).cuda()
    # # sdnet = snDiscriminator().cuda()
    # if args.gnet:
    #     sgnet.load_state_dict(torch.load(args.gnet))
    # if args.dnet:
    #     sdnet.load_state_dict(torch.load(args.dnet))

    optimizerD = optim.SGD(Dnet.parameters(), lr=args.lr*4, momentum=0.9,
                          weight_decay=5e-4)
    optimizerG = optim.SGD(Gnet.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4)

    # optimizerD = optim.Adam(filter(lambda p: p.requires_grad, sdnet.parameters()), lr=args.lr*4,betas=(0.5,0.999))
    # optimizerG = optim.Adam(sgnet.parameters(),lr=args.lr,betas=(0.5,0.999))
    criterion = nn.SmoothL1Loss().cuda()

    print('Loading the dataset...')
    data = Pokédex()
    PokéBall = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print('Training GhostNet on:', data.name)
    print('Using the specified args:')
    print(args)

    # Dnet.eval()
    Dnet.train()
    Gnet.train()
    # create batch iterator
    for iteration in range(args.start_iter + 1, args.epochs):
        print('Epoch %d' % iteration, end=' ')
        train_encoder_decoder(PokéBall, Dnet, Gnet, criterion, optimizerD, optimizerG)
        # sdnet.train()
        # sgnet.eval()
        # loss_d = train_dnet(PokéBall, Dnet, sgnet, sdnet, optimizerD)
        # sgnet.train()
        # sdnet.eval()
        # for i in range(5):
        #     loss_g = train_gnet(PokéBall, sgnet, sdnet, optimizerG)
        data.shuffle()
        if not (iteration-args.start_iter) == 0 and iteration % 50 == 0:
            torch.save(Gnet.state_dict(),
                        ops('checkpoints', 'd', 'Pokemonet_d.pth%03d' % iteration))
            torch.save(Gnet.state_dict(),
                        ops('checkpoints', 'g', 'Pokemonet_g.pth%03d' % iteration))
    torch.save(Gnet.state_dict(),
                ops('checkpoints', 'd', 'Pokemonet_d.pth%d' % args.epochs))
    torch.save(Gnet.state_dict(),
                ops('checkpoints', 'g', 'Pokemonet_g.pth%d' % args.epochs))
if __name__ == '__main__':
    train()