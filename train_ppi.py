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
from utils.preprocessing import load_ppi_data, str_to_list
from utils.ppi_data import PpiData
from models import GAT, SpGAT, SpGAT_inductive, GCN

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
parser.add_argument('--hidden', type=str, default='8',
                    help='Number of hidden units.')
parser.add_argument('--nb_heads', type=str, default='8',
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
n_heads = str_to_list(args.nb_heads, dtype=int)
n_hidden = str_to_list(args.hidden, dtype=int)
# Model and optimizer
if args.sparse:
    model = SpGAT_inductive(nfeat=n_feat,
                  nhid=n_hidden,
                  nclass=n_classes,
                  dropout=args.dropout,
                  nheads=n_heads,
                  alpha=args.alpha)
else:
    model = GAT(nfeat=n_feat,
                nhid=n_hidden,
                nclass=n_classes,
                dropout=args.dropout,
                nheads=n_heads,
                alpha=args.alpha)

model = GCN(n_feat, n_classes, args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    ppi_data.to_device()

n_train = ppi_data.train_adj.shape[0]
n_val = ppi_data.val_adj.shape[0]
n_test = ppi_data.test_adj.shape[0]
n_nodes = ppi_data.train_adj.shape[1]
f1_threshold = 0.5


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
        loss_train += F.binary_cross_entropy(output, target_labels.float())
        f1_train += fbeta_score(output, target_labels, threshold=f1_threshold)
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
        loss_val += F.binary_cross_entropy(output, target_labels.float())
        f1_val += fbeta_score(output, target_labels, threshold=f1_threshold)

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
    model.eval()
    loss_test = 0.
    f1_test = 0.

    for i in range(n_test):
        test_mask = ppi_data.ts_msk[i].byte()
        output = model(ppi_data.test_feat[i], ppi_data.test_adj[i])[test_mask, ]
        target_labels = ppi_data.test_labels[i, test_mask]
        loss_test += F.multilabel_margin_loss(output, target_labels)
        f1_test += fbeta_score(torch.exp(output), target_labels, threshold=f1_threshold)

    loss_test /= n_test
    f1_test /= n_test

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "F1-test= {:.4f}".format(f1_test.item()))


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
compute_test()
