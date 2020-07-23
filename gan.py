from Pokedex import Pokédex
from resnet import getDiscriminator, getGenerator

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
parser.add_argument('--gnet', type=str, default=None,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--dnet', type=str, default=None,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--epochs', '-e', default=400, type=int,
                    help='the number of training epochs')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
args = parser.parse_args()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train_dnet(loader, dnet, gnet, criterion, optimizer):
    loss_amount = 0
    # load train data
    for iteration, batch in enumerate(loader):
        # forward & backprop
        optimizer.zero_grad()
        
        Pokémon = batch.permute(0,3,1,2).type(torch.float).cuda()
        size = Pokémon.size()[0]
        fakePoké = gnet(torch.randn(size*2,256,4,4))

        shuffle = torch.randperm(size*2)
        samples = torch.cat([fakePoké, Pokémon], 0)[shuffle]
        labels = torch.cat([torch.zeros(size), torch.ones(size)], 0)[shuffle]
        preds = dnet(samples)
        loss = criterion(preds, labels.type(torch.long))
        loss.backward()
        optimizer.step()
        loss_amount += loss.item()
    print('Dnet Loss: %.6f ' % (loss_amount/iteration))
    return loss_amount/iteration

def train_gnet(length, dnet, gnet, criterion, optimizer, size):
    loss_amount = 0
    # load train data
    for iteration in range(length):
        # forward & backprop
        optimizer.zero_grad()
        
        fakePoké = gnet(torch.randn(size*2,256,4,4))
        labels = torch.ones(size*2)
        preds = dnet(fakePoké)
        loss = criterion(preds, labels.type(torch.long))
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
        _, cores = dnet(Pokémon)
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

    if args.gnet:
        missing, unexpected = Gnet.load_state_dict(torch.load(args.gnet))
        if missing:
            print('Missing:', missing)
        if unexpected:
            print('Unexpected:', unexpected)
    if args.dnet:
        missing, unexpected = Dnet.load_state_dict(torch.load(args.dnet))
        if missing:
            print('Missing:', missing)
        if unexpected:
            print('Unexpected:', unexpected)

    optimizerD = optim.RMSprop(Dnet.parameters(), lr=args.lr*4)
    optimizerG = optim.RMSprop(Gnet.parameters(), lr=args.lr)
    # optimizerD = optim.Adam(Dnet.parameters(),lr=args.lr*4,betas=(0.5,0.999))
    # optimizerG = optim.Adam(Gnet.parameters(),lr=args.lr,betas=(0.5,0.999))
    # for param_group in optimizer.param_groups:
    #     param_group['initial_lr'] = args.lr
    # adjust_learning_rate = optim.lr_scheduler.MultiStepLR(optimizer, [35, 55], 0.1, args.start_iter)
    # adjust_learning_rate = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.start_iter)
    criterion = nn.SmoothL1Loss().cuda()

    
    print('Loading the dataset...')
    data = Pokédex()
    PokéBall = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print('Training GhostNet on:', data.name)
    print('Using the specified args:')
    print(args)

    # create batch iterator
    for iteration in range(args.start_iter + 1, args.epochs):
        print('Epoch %d' % iteration, end=' ')
        Dnet.train()
        Gnet.train()
        train_encoder_decoder(PokéBall, Dnet, Gnet, criterion, optimizerD, optimizerG)
        # Dnet.train()
        # Gnet.eval()
        # for i in range(1):
        #     loss_d = train_dnet(PokéBall, Dnet, Gnet, criterion, optimizerD)

        # Gnet.train()
        # Dnet.eval()
        # for i in range(3):
        #     loss_g = train_gnet(len(PokéBall), Dnet, Gnet, criterion, optimizerG, args.batch_size)
        # adjust_learning_rate.step()
        if not (iteration-args.start_iter) == 0 and iteration % 10 == 0:
            torch.save(Dnet.state_dict(),
                        ops('checkpoints', 'd', 'Pokemonet_d.pth%03d' % iteration))
            torch.save(Gnet.state_dict(),
                        ops('checkpoints', 'g', 'Pokemonet_g.pth%03d' % iteration))
    torch.save(Dnet.state_dict(),
                ops('checkpoints', 'd', 'Pokemonet_%d_d.pth' % args.epochs))
    torch.save(Gnet.state_dict(),
                ops('checkpoints', 'g', 'Pokemonet_%d_g.pth' % args.epochs))
if __name__ == '__main__':
    train()