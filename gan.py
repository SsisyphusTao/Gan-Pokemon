from Pokedex import Pokédex
from resnet import getDiscriminator, getGenerator, lasthope

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
parser.add_argument('--gnet', type=str, default='checkpoints/Pokemonet_g.pth',
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--dnet', type=str, default='checkpoints/Pokemonet_d.pth',
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--epochs', '-e', default=400, type=int,
                    help='the number of training epochs')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
args = parser.parse_args()
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def train_dnet(loader, dnet, gnet, cnet, netd, criterion, optimizer):
    loss_amount = 0
    # load train data
    for iteration, batch in enumerate(loader):
        # forward & backprop
        optimizer.zero_grad()
        for p in netd.parameters():
            p.data.clamp_(-0.01, 0.01)
        
        Pokémon = batch.permute(0,3,1,2).type(torch.float).cuda()
        size = Pokémon.size()[0]

        core = dnet(Pokémon)
        # realPoké = gnet(core)

        Pokécore = cnet(torch.randn(size,3,128,128))
        # fakePoké = gnet(Pokécore)

        preds_r = netd(core)
        one = torch.randn_like(preds_r).clamp_(-2, 2)+1
        loss_r = criterion(preds_r, one)

        preds_f = netd(Pokécore)
        mone = torch.randn_like(preds_f).clamp_(-2, 2)-1
        loss_f = criterion(preds_f, mone)

        loss = loss_r + loss_f
        loss.backward()
        optimizer.step()
        loss_amount += loss.item()
    print('Dnet Loss: %.6f ' % (loss_amount/iteration))
    return loss_amount/iteration

def train_gnet(loader, gnet, cnet, netd, criterion, optimizer):
    loss_amount = 0
    # load train data
    for iteration, batch in enumerate(loader):
        # forward & backprop
        optimizer.zero_grad()

        Pokémon = batch.permute(0,3,1,2).type(torch.float).cuda()
        size = Pokémon.size()[0]
        
        Pokécore = cnet(torch.randn(size*2,3,128,128))
        # core = dnet(Pokémon)
        # fakePoké = gnet(Pokécore)
        
        preds = netd(Pokécore)
        one = one = torch.randn_like(preds).clamp_(-1, 1)+1
        loss = criterion(preds, one)
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
    Cnet = getDiscriminator(18).cuda()
    netD = lasthope().cuda()
    if args.gnet:
        Gnet.load_state_dict(torch.load(args.gnet))
    if args.dnet:
        Dnet.load_state_dict(torch.load(args.dnet), strict=False)
        Cnet.load_state_dict(torch.load(args.dnet), strict=False)

    optimizerD = optim.RMSprop(netD.parameters(), lr=args.lr*4)
    optimizerC = optim.RMSprop(Cnet.parameters(), lr=args.lr)
    # optimizerD = optim.Adam(Dnet.parameters(),lr=args.lr*4,betas=(0.5,0.999))
    # optimizerG = optim.Adam(Gnet.parameters(),lr=args.lr,betas=(0.5,0.999))
    # optimizerC = optim.Adam(Cnet.parameters(),lr=args.lr,betas=(0.5,0.999))
    # for param_group in optimizer.param_groups:
    #     param_group['initial_lr'] = args.lr
    # adjust_learning_rate = optim.lr_scheduler.MultiStepLR(optimizer, [35, 55], 0.1, args.start_iter)
    # adjust_learning_rate = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.start_iter)
    criterion = nn.MSELoss().cuda()

    print('Loading the dataset...')
    data = Pokédex()
    PokéBall = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print('Training GhostNet on:', data.name)
    print('Using the specified args:')
    print(args)
    Gnet.eval()
    Dnet.eval()
    # create batch iterator
    for iteration in range(args.start_iter + 1, args.epochs):
        print('Epoch %d' % iteration, end=' ')
        # Dnet.train()
        # Gnet.train()
        # train_encoder_decoder(PokéBall, Dnet, Gnet, criterion, optimizerD, optimizerG)
        netD.train()
        Cnet.eval()
        loss_d = train_dnet(PokéBall, Dnet, Gnet, Cnet, netD, criterion, optimizerD)

        Cnet.train()
        netD.eval()
        for i in range(2):
            loss_g = train_gnet(PokéBall, Gnet, Cnet, netD, criterion, optimizerC)
        if not (iteration-args.start_iter) == 0 and iteration % 10 == 0:
            torch.save(Cnet.state_dict(),
                        ops('checkpoints', 'd', 'Pokemonet_d.pth%03d' % iteration))
            # torch.save(Gnet.state_dict(),
            #             ops('checkpoints', 'g', 'Pokemonet_g.pth%03d' % iteration))
    torch.save(Cnet.state_dict(),
                ops('checkpoints', 'd', 'Pokemonet_d.pth%d' % args.epochs))
    # torch.save(Gnet.state_dict(),
    #             ops('checkpoints', 'g', 'Pokemonet_g.pth%d' % args.epochs))
if __name__ == '__main__':
    train()