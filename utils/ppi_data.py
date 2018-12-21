class PpiData:
    def __init__(self, train_adj,val_adj,test_adj,train_feat,val_feat,test_feat,train_labels,val_labels,
     test_labels, train_nodes, val_nodes, test_nodes, tr_msk, vl_msk, ts_msk):
        self.train_adj = train_adj
        self.val_adj = val_adj
        self.test_adj = test_adj
        self.train_feat = train_feat
        self.val_feat = val_feat
        self.test_feat = test_feat
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes
        self.tr_msk = tr_msk
        self.vl_msk = vl_msk
        self.ts_msk = ts_msk

    def to_device(self, device=None):
        raise NotImplementedError()

    def save(self, path='./data'):
        raise NotImplementedError()