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
parser.add_argument('task', default='d',
                    help='d | g')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--gnet', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--dnet', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--epochs', default=70, type=int,
                    help='the number of training epochs')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
args = parser.parse_args()
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

# @profile
def train_one_epoch(loader, dnet, gnet, criterion, optimizer, epoch, size):
    loss_amount = 0
    # load train data
    for iteration, batch in enumerate(loader):
        # forward & backprop
        optimizer.zero_grad()
        
        shuffle = torch.randperm(size*2)
        fakePoké = gnet(torch.randn(size,64,8,8))
        Pokémon = batch.permute(0,3,1,2).type(torch.float)
        samples = torch.cat([fakePoké, Pokémon], 0)[shuffle]
        labels = torch.cat([torch.zeros(size), torch.ones(size)], 0)[shuffle]
        preds = dnet(samples)
        loss = criterion(preds, labels.type(torch.long))
        loss.backward()
        optimizer.step()
        loss_amount += loss.item()
        if iteration % 10 == 0 and not iteration == 0:
            print('Loss: %.6f, %.6f | iter: %03d | epoch: %d' %
                    (loss_amount/iteration, loss.item(), iteration, epoch))
    print('Loss: %.6f on epoch %d -----------------------' % (loss_amount/iteration, epoch))
    return loss_amount/iteration*1000

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

    if args.task == 'd':
        Dnet.train()
        Gnet.eval()
        optimizer = optim.SGD(Dnet.parameters(), lr=args.lr, momentum=0.9,
                        weight_decay=5e-4)
    elif args.task == 'g':
        Gnet.train()
        Dnet.eval()
        optimizer = optim.SGD(Gnet.parameters(), lr=args.lr, momentum=0.9,
                        weight_decay=5e-4)

    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.lr
    adjust_learning_rate = optim.lr_scheduler.MultiStepLR(optimizer, [35, 55], 0.1, args.start_iter)
    # adjust_learning_rate = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.start_iter)
    # getloss = nn.parallel.DistributedDataParallel(NetwithLoss(net), device_ids=[args.local_rank], find_unused_parameters=True)
    criterion = torch.nn.CrossEntropyLoss()

    
    print('Loading the dataset...')
    data = Pokédex()
    PokéBall = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print('Training GhostNet on:', data.name)
    print('Using the specified args:')
    print(args)

    # create batch iterator
    for iteration in range(args.start_iter + 1, args.epochs):
        loss = train_one_epoch(PokéBall, Dnet, Gnet, criterion, optimizer, iteration, args.batch_size)
        adjust_learning_rate.step()
        if not (iteration-args.start_iter) == 0:
            torch.save(Dnet.state_dict() if args.task == 'd' else Gnet.state_dict(),
                        ops(args.save_folder, args.task, 'Pokémonet_%03d_%d.pth' % (iteration, loss)))
        torch.save(Dnet.state_dict() if args.task == 'd' else Gnet.state_dict(),
                    ops(args.save_folder, args.task, 'Pokémonet_%03d_%d.pth' % (args.epochs, loss)))
if __name__ == '__main__':
    train()