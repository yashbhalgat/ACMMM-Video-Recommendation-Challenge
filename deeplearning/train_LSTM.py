from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tripletnetwork import Tripletnet, LSTMNet, FusedLSTMNet
import numpy as np
import pickle
import random
import csv
from numpy import genfromtxt

from utils import get_video_features, get_features

import pdb

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=1.0, metavar='M',
                    help='margin for triplet loss (default: 1.0)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='LSTMNet', type=str,
                    help='name of experiment')
parser.add_argument('--similarity', default='softmax', type=str,
                    help='name of similarity measure')
parser.add_argument('--embeddsize', type=int, default=256, metavar='M',
                    help='embedding size (default: 256)')

best_acc = 0

main_dir = "/scratch/jiadeng_fluxoe/yashsb/ACMMM_challenge/release/release/"

shows_dir = main_dir + "track_1_shows/"
movies_dir = main_dir + "track_2_movies/"


def load_gt(shows_dir, movies_dir):
    ## Shows gt 
    shows_train_gt = []
    fp = open(shows_dir+"relevance_train.csv", "r")
    for row in csv.reader(fp):
        row = [i for i in row if i.isdigit()]
        shows_train_gt.append(row)
    fp.close()

    shows_val_gt = []
    fp = open(shows_dir+"relevance_val.csv", "r")
    for row in csv.reader(fp):
        row = [i for i in row if i.isdigit()]
        shows_val_gt.append(row)
    fp.close()

    ## Movies gt 
    movies_train_gt = []
    fp = open(movies_dir+"relevance_train.csv", "r")
    for row in csv.reader(fp):
        row = [i for i in row if i.isdigit()]
        movies_train_gt.append(row)
    fp.close()

    movies_val_gt = []
    fp = open(movies_dir+"relevance_val.csv", "r")
    for row in csv.reader(fp):
        row = [i for i in row if i.isdigit()]
        movies_val_gt.append(row)
    fp.close()

    return shows_train_gt, shows_val_gt, movies_train_gt, movies_val_gt


def load_set(shows_dir, movies_dir, phase="test"):
    # loading test set
    shows_set = genfromtxt(shows_dir+"split/"+phase+".csv", delimiter=',', dtype=str)
    shows_set = list(shows_set)
    shows_set = [int(i) for i in shows_set]
    movies_set = genfromtxt(movies_dir+"split/"+phase+".csv", delimiter=',', dtype=str)
    movies_set = list(movies_set)
    movies_set = [int(i) for i in movies_set]

    return shows_set, movies_set


def get_norm_params(features):
    '''
    Usage: features = shows_features
    '''
    feat_length = features[0].shape[0]
    num_samples = len(features.keys())

    mean_vec = np.zeros((feat_length))
    for ind in range(feat_length):
        sum = 0
        for c in features.keys():
            sum += features[c][ind]
        mean_vec[ind] = sum/num_samples

    std_vec = np.zeros((feat_length))
    for ind in range(feat_length):
        std_vec[ind] = np.std([features[c][ind] for c in features.keys()])

    params = {}
    params["mean_vec"] = mean_vec
    params["std_vec"] = std_vec

    return params

def normalize_features(features, params):   # features = shows_features
    mean_vec = params["mean_vec"]
    std_vec = params["std_vec"]

    for c in features.keys():
        features[c] = (features[c]-mean_vec)/std_vec

    return features

def main():
    global args, best_acc
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    generate_triplets = True
    if generate_triplets:
        shows_train_gt, shows_val_gt, movies_train_gt, movies_val_gt = load_gt(shows_dir, movies_dir)
        shows_valid_set, movies_valid_set = load_set(shows_dir, movies_dir, "val")
        train_size = len(shows_train_gt)
        shows_set = [str(i) for i in range(7536)]
        triplets = []
        
        ## Generate triplets
        state = np.random.RandomState(None)
        print("PHASE 1")
        for iteration in range(33200):
            try:
                p_ind = state.randint(train_size)

                n_rel = len(shows_train_gt[p_ind])-1
                if n_rel < 2:
                    continue

                pos_ind = state.randint(n_rel-1)+1
                neg_ind = state.randint(n_rel-pos_ind)+pos_ind+1

                p_vid = int(shows_train_gt[p_ind][0])
                pos_vid = int(shows_train_gt[p_ind][pos_ind])
                neg_vid = int(shows_train_gt[p_ind][neg_ind])

                # print(p_vid, pos_vid, neg_vid)
                triplets.append((p_vid, pos_vid, neg_vid))
            except:
                pass

        print("PHASE 2")
        for num_times in range(8):
            for t in range(len(shows_train_gt)):
                try:
                    rel_vids = [int(r) for r in shows_train_gt[t]]
                    if len(rel_vids) < 2:
                        continue

                    for sub_iter in range(8):
                        p_ind = state.randint(len(rel_vids))
                        pos_ind = state.randint(len(rel_vids))
                        while pos_ind==p_ind:
                            pos_ind = state.randint(len(rel_vids))

                        p_vid = rel_vids[p_ind]
                        pos_vid = rel_vids[pos_ind]

                        for sub_sub in range(8):
                            neg_vid = state.randint(len(shows_set))
                            while neg_vid in rel_vids:
                                neg_vid = state.randint(len(shows_set))

                            # print(p_vid, pos_vid, neg_vid)
                            triplets.append((p_vid, pos_vid, neg_vid))
                except:
                    pass

        # triplets = triplets1*8 + triplets2*8
        random.shuffle(triplets)
        np.save("triplets.npy", triplets)
    else:
        triplets = np.load("triplets.npy")

    shows_set = [str(i) for i in range(7536)]
    movies_set = []
    shows_features, _ = get_video_features(shows_set, shows_dir, reload_features=False)
    shows_c3d, _ = get_features(shows_set, movies_set, shows_dir, movies_dir, feature_type="c3d", reload_features=False)

    # params = get_norm_params(shows_features)
    # shows_features = normalize_features(shows_features, params)

    # net = FusedLSTMNet(args.embeddsize)
    net = LSTMNet(args.embeddsize)
    tnet = Tripletnet(net)
    
    if args.cuda:
        tnet.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    cudnn.benchmark = True

    margin = args.margin

    criterion = torch.nn.MarginRankingLoss(margin = args.margin)

    optimizer = optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    # pdb.set_trace()

    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        train(triplets, shows_features, tnet, criterion, optimizer, epoch, args.start_epoch, margin, features_c3d=None)
        # evaluate on validation set
        # acc = test(test_loader, tnet, criterion, epoch)

        # # remember best acc and save checkpoint
        # is_best = acc > best_acc
        # best_acc = max(acc, best_acc)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': tnet.state_dict(),
        # }, True)

def train(triplets, features, tnet, criterion, optimizer, epoch, start_iter, margin, features_c3d=None):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()

    # switch to train mode
    tnet.train()
    for batch_idx, (p, p_plus, p_minus) in enumerate(triplets):
        data1 = torch.from_numpy(features[p]).float().unsqueeze(0)
        data2 = torch.from_numpy(features[p_plus]).float().unsqueeze(0)
        data3 = torch.from_numpy(features[p_minus]).float().unsqueeze(0)
        
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        if features_c3d!=None:
            data1_c3d = torch.from_numpy(features_c3d[p]).float().unsqueeze(0)
            data2_c3d = torch.from_numpy(features_c3d[p_plus]).float().unsqueeze(0)
            data3_c3d = torch.from_numpy(features_c3d[p_minus]).float().unsqueeze(0)

            if args.cuda:
                data1_c3d, data2_c3d, data3_c3d = data1_c3d.cuda(), data2_c3d.cuda(), data3_c3d.cuda()
            data1_c3d, data2_c3d, data3_c3d = Variable(data1_c3d), Variable(data2_c3d), Variable(data3_c3d)

        # compute output
        if features_c3d!=None:
            sim_a, sim_b, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3, data1_c3d, data2_c3d, data3_c3d, similarity=args.similarity)
        else:
            sim_a, sim_b, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3, similarity=args.similarity)
        # 1 means, dista should be larger than distb
        # pdb.set_trace()

        if args.similarity=="distance":
            target = torch.FloatTensor(sim_a.size()).fill_(-1)
            factor = -1
        else:
            target = torch.FloatTensor(sim_a.size()).fill_(1)
            factor = 1

        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        
        #### CAREFUL!!!!
        loss_triplet = criterion(sim_a, sim_b, target)
        # loss_triplet = (sim_a-1)**2 + sim_b**2
        

        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.0001 * loss_embedd
        # loss_embedd = 0

        # measure accuracy and record loss
        acc = accuracy(sim_a, sim_b, margin, factor)
        losses.update(loss_triplet.data[0], 1)
        accs.update(acc, 1)
        emb_norms.update(loss_embedd.data[0]/3, 1)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * 1, len(triplets),
                losses.val, losses.avg, 
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))

        if batch_idx%100000==0 and batch_idx>0:
            save_checkpoint({
                'epoch': batch_idx + start_iter,
                'state_dict': tnet.state_dict(),
            }, True)
    
def test(test_loader, tnet, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, (data1, data2, data3) in enumerate(test_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, _, _, _ = tnet(data1, data2, data3)
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        test_loss =  criterion(dista, distb, target).data[0]

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        accs.update(acc, data1.size(0))
        losses.update(test_loss, data1.size(0))      

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    plotter.plot('acc', 'test', epoch, accs.avg)
    plotter.plot('loss', 'test', epoch, losses.avg)
    return accs.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = 'checkpoint_'+str(state["epoch"])+'.pth.tar'

    filename = directory + filename
    

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(dista, distb, margin, factor=1):
    # margin = 0
    pred = (dista - distb - margin).cpu().data
    if factor==-1:
        pred = (distb - dista - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]

if __name__ == '__main__':
    main()    
