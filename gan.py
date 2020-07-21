from Pokedex import Pokédex
from resnet import get_pose_net

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import time

parser = argparse.ArgumentParser(
    description="Pokémon, Getto Daze!")
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', type=str,
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
def train_one_epoch(loader, net, criterion, optimizer, epoch):
    loss_amount = 0
    # load train data
    for iteration, batch in enumerate(loader):
        # forward & backprop
        optimizer.zero_grad()
        preds = net(batch.permute(0,3,1,2).type(torch.float))
        loss = criterion(preds['cls'], torch.zeros(preds['cls'].size()[0],preds['cls'].size()[2],preds['cls'].size()[3]).type(torch.long))
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
    heads = {'cls':2}
    net = get_pose_net(
        18, heads, 64
    )
    if args.resume:
        missing, unexpected = net.load_state_dict(torch.load(args.resume))
        if missing:
            print('Missing:', missing)
        if unexpected:
            print('Unexpected:', unexpected)
    net.train()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
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
        loss = train_one_epoch(PokéBall, net, criterion, optimizer, iteration)
        adjust_learning_rate.step()
        if not (iteration-args.start_iter) == 0:
            torch.save(net.state_dict(), args.save_folder + 'Pokémonet_%03d_%d.pth' % (iteration, loss))
        torch.save(net.state_dict(),
                    args.save_folder + 'Pokémonet_%d_%d.pth' % (iteration, loss))
if __name__ == '__main__':
    train()