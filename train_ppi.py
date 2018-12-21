from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils.preprocessing import load_data, accuracy, load_ppi_data
from utils.ppi_data import PpiData
from models import GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true',
                    default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true',
                    default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8,
                    help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()

ppi_data = load_ppi_data('/Users/kaumad/Documents/coding/data/network_data/ppi/toy_data')
n_feat = ppi_data.train_feat[0].shape[1]
n_classes = ppi_data.train_labels.shape[2]

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=n_feat,
                  nhid=args.hidden,
                  nclass=n_classes,
                  dropout=args.dropout,
                  nheads=args.nb_heads,
                  alpha=args.alpha)
else:
    model = GAT(nfeat=n_feat,
                nhid=args.hidden,
                nclass=n_classes,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    ppi_data.to_device()
    # features = features.cuda()
    # adj = adj.cuda()
    # labels = labels.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()

# features, adj, labels = Variable(features), Variable(adj), Variable(labels)
train_adj, train_feat, train_labels = Variable(ppi_data.train_adj), Variable(
    ppi_data.train_feat), Variable(ppi_data.train_labels)
val_adj, val_feat, val_labels = Variable(ppi_data.val_adj), Variable(
    ppi_data.val_feat), Variable(ppi_data.val_labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    loss_train = 0.
    acc_train = 0.
    n_train = train_adj.shape[0]

    for i in range(n_train):
        output = model(train_feat[i], train_adj[i])
        loss_train += F.nll_loss(output, train_labels[i])
        acc_train += accuracy(output, train_labels[i])

    loss_train /= n_train
    acc_train /= n_train

    # output = model(features, adj)
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    loss_val = 0.
    acc_val = 0.

    for i in range(len(val_adj)):
        output = model(val_feat[i], val_adj[i])
        loss_val += F.nll_loss(output, val_labels[i])
        acc_val += accuracy(output, val_labels[i])

    loss_val /= len(train_adj)
    acc_val /= len(train_adj)

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.item()


def compute_test():
    raise NotImplementedError()
    # model.eval()
    # output = model(features, adj)
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    # acc_test = accuracy(output[idx_test], labels[idx_test])
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
# TODO fix testing code
# compute_test()
