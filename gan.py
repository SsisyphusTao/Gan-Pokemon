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
parser.add_argument('--gnet', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--dnet', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--epochs', '-e', default=50, type=int,
                    help='the number of training epochs')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
args = parser.parse_args()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train_dnet(loader, dnet, gnet, criterion, optimizer, epoch):
    loss_amount = 0
    # load train data
    for iteration, batch in enumerate(loader):
        # forward & backprop
        optimizer.zero_grad()
        
        Pokémon = batch.permute(0,3,1,2).type(torch.float).cuda()
        size = Pokémon.size()[0]
        fakePoké = gnet(torch.randn(size,64,8,8))

        shuffle = torch.randperm(size*2)
        samples = torch.cat([fakePoké, Pokémon], 0)[shuffle]
        labels = torch.cat([torch.zeros(size), torch.ones(size)], 0)[shuffle]
        preds = dnet(samples)
        loss = criterion(preds, labels.type(torch.long))
        loss.backward()
        optimizer.step()
        loss_amount += loss.item()
        # if iteration % 10 == 0 and not iteration == 0:
        #     print('Loss: %.6f, %.6f | iter: %03d | epoch: %d' %
        #             (loss_amount/iteration, loss.item(), iteration, epoch))
    print('Dnet Loss: %.6f ' % (loss_amount/iteration))
    return loss_amount/iteration

def train_gnet(length, dnet, gnet, criterion, optimizer, epoch, size):
    loss_amount = 0
    # load train data
    for iteration in range(length):
        # forward & backprop
        optimizer.zero_grad()
        
        fakePoké = gnet(torch.randn(size*2,64,8,8))
        labels = torch.ones(size*2)
        preds = dnet(fakePoké)
        loss = criterion(preds, labels.type(torch.long))
        loss.backward()
        optimizer.step()
        loss_amount += loss.item()
        # if iteration % 10 == 0 and not iteration == 0:
        #     print('Loss: %.6f, %.6f | iter: %03d | epoch: %d' %
        #             (loss_amount/iteration, loss.item(), iteration, epoch))
    print('   Gnet Loss: %.6f' % (loss_amount/iteration))
    return loss_amount/iteration

def train():

    torch.backends.cudnn.benchmark = True
    Gnet = getGenerator()
    Dnet = getDiscriminator(18)

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

    # for param_group in optimizer.param_groups:
    #     param_group['initial_lr'] = args.lr
    # adjust_learning_rate = optim.lr_scheduler.MultiStepLR(optimizer, [35, 55], 0.1, args.start_iter)
    # adjust_learning_rate = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.start_iter)
    criterion = torch.nn.CrossEntropyLoss()

    
    print('Loading the dataset...')
    data = Pokédex()
    PokéBall = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print('Training GhostNet on:', data.name)
    print('Using the specified args:')
    print(args)

    # create batch iterator
    for iteration in range(args.start_iter + 1, args.epochs):
        print('Epoch: %d -------------' % iteration)
        Dnet.train()
        Gnet.eval()
        optimizer = optim.SGD(Dnet.parameters(), lr=args.lr, momentum=0.9,
                        weight_decay=5e-4)
        for i in range(2):
            loss_d = train_dnet(PokéBall, Dnet.cuda(), Gnet.cuda(), criterion.cuda(), optimizer, int(iteration))

        Gnet.train()
        Dnet.eval()
        optimizer = optim.SGD(Gnet.parameters(), lr=args.lr, momentum=0.9,
                        weight_decay=5e-4)
        for i in range(5):
            loss_g = train_gnet(len(PokéBall), Dnet.cuda(), Gnet.cuda(), criterion.cuda(), optimizer, int(iteration), args.batch_size)
        # adjust_learning_rate.step()
        if not (iteration-args.start_iter) == 0 and iteration % 10 == 0:
            torch.save(Dnet.state_dict(),
                        ops('checkpoints', 'd', 'Pokemonet_%d_%03d_%.3e.pth' % (args.epochs, iteration, loss_d)))
            torch.save(Gnet.state_dict(),
                        ops('checkpoints', 'g', 'Pokemonet_%d_%03d_%.3e.pth' % (args.epochs, iteration, loss_g)))
    torch.save(Dnet.state_dict(),
                ops('checkpoints', 'd', 'Pokemonet_%d_%.3e.pth' % (args.epochs, loss_d)))
    torch.save(Gnet.state_dict(),
                ops('checkpoints', 'g', 'Pokemonet_%d_%.3e.pth' % (args.epochs, loss_g)))
if __name__ == '__main__':
    train()