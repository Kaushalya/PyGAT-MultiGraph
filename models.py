import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer, GraphLinearLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(
            nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(
            nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return torch.sigmoid(x)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return torch.sigmoid(x)


class SpGAT_inductive(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT with three Graph Attention layers."""
        super(SpGAT_inductive, self).__init__()
        self.dropout = dropout
        self.n_classes = nclass
        if len(nheads) == 1:
            nheads = [nheads[0], nheads[0], 1]
        if len(nhid) == 1:
            nhid = [nhid[0], nhid[0]]
        self.attentions1 = [SpGraphAttentionLayer(nfeat,
                                                  nhid[0],
                                                  dropout=dropout,
                                                  alpha=alpha,
                                                  concat=True) for _ in range(nheads[0])]
        self.attentions2 = [SpGraphAttentionLayer(nhid[0] * nheads[0],
                                                  nhid[1],
                                                  dropout=dropout,
                                                  alpha=alpha,
                                                  concat=True) for _ in range(nheads[1])]
        self.out_attentions = [SpGraphAttentionLayer(nhid[1] * nheads[1],
                                                     nclass,
                                                     dropout=dropout,
                                                     alpha=alpha,
                                                     concat=False) for _ in range(nheads[2])]
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention)
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)
        for i, attention in enumerate(self.out_attentions):
            self.add_module('out_attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions1], dim=1)
        x = torch.cat([att(x, adj) for att in self.attentions2], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj).view(-1, self.n_classes, 1) for
                       att in self.out_attentions], dim=2)
        logits = torch.mean(x, dim=2)
        return torch.sigmoid(logits)


class GCN(nn.Module):
    def __init__(self, n_feat, n_classes, dropout, ch_list=None):
        """Implementation of Graph Convolutional Networks (Kipf and Welling, ICLR 2017)"""
        super(GCN, self).__init__()
        ch_list = ch_list or [n_feat, 256, 128]
        self.dropout = dropout
        self.gconvs = [GraphLinearLayer(ch_list[i], ch_list[i+1]) for i in range(len(ch_list)-1)]
        self.lin = GraphLinearLayer(ch_list[-1], n_classes)
        for i, gconv in enumerate(self.gconvs):
            self.add_module('gconv_{}'.format(i), gconv)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        for gconv in self.gconvs:
            x = F.relu(gconv(x, adj))
        h = self.lin(x, adj)
        return torch.sigmoid(h)
