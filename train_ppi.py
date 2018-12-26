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

from utils.metrics import fbeta_score
from utils.preprocessing import load_ppi_data
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
parser.add_argument('--data_dir', type=str, default='./data/ppi-toy-data')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# load data
ppi_data = load_ppi_data(args.data_dir)
n_feat = ppi_data.train_feat.shape[2]
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

n_train = ppi_data.train_adj.shape[0]
n_val = ppi_data.val_adj.shape[0]
n_nodes = ppi_data.train_adj.shape[1]


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    loss_train = 0.
    # acc_train = 0.
    f1_train = 0.

    for i in range(n_train):
        # optimizer.zero_grad()
        node_mask = ppi_data.tr_msk[i].byte()
        output = model(ppi_data.train_feat[i], ppi_data.train_adj[i])[node_mask, ]
        target_labels = ppi_data.train_labels[i, node_mask]
        loss_train += F.multilabel_soft_margin_loss(output, target_labels.float())
        f1_train += fbeta_score(torch.exp(output), target_labels, threshold=0.00827)
        # loss_train.backward()
        # optimizer.step()

    loss_train /= n_train
    f1_train /= n_train

    loss_train.backward()
    optimizer.step()

    # calculation of validation loss
    loss_val = 0.
    f1_val = 0.

    for i in range(len(ppi_data.val_adj)):
        val_mask = ppi_data.vl_msk[i].byte()
        output = model(ppi_data.val_feat[i], ppi_data.val_adj[i])[val_mask, ]
        target_labels = ppi_data.val_labels[i, val_mask]
        loss_val += F.multilabel_margin_loss(output, target_labels)
        f1_val += fbeta_score(torch.exp(output), target_labels, threshold=0.00827)

    loss_val /= n_val
    f1_val /= n_val

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'f1_train: {:.4f}'.format(f1_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'f1_val: {:.4f}'.format(f1_val.item()),
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
