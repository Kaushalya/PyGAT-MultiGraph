import torch


def accuracy(preds, targs):
    preds = preds.max(1)[1].type_as(targs)
    correct = preds.eq(targs).double()
    correct = correct.sum()
    return correct / len(targs)


def accuracy_multilabel(preds, targs, threshold=0.5):

    raise NotImplementedError()


def fbeta_score(preds, targs, beta=1., threshold=0.5):
    prec = precision(preds, targs, threshold=threshold)
    rec = recall(preds, targs, threshold=threshold)
    beta2 = beta * beta
    num = (1 + beta2) * prec * rec
    denom = beta2 * (prec + rec)

    if denom == 0:
        return 0.

    return num/denom


def recall(preds, targs, threshold=0.5):
    pred_pos = preds > threshold
    true_pos = (pred_pos[pred_pos] == targs[pred_pos].byte()).sum()
    return true_pos/targs.float().sum()


def precision(preds, targs, threshold=0.5):
    pred_pos = preds > threshold
    true_pos = (pred_pos[pred_pos] == targs[pred_pos].byte()).sum()
    return true_pos/pred_pos.float().sum()


if __name__ == '__main__':
    # for testing implementation
    preds = torch.tensor([0.3, 0.51, 0.6, 0.1, 0.4])
    targs = torch.tensor([1, 1, 0, 0, 0])

    rec = recall(preds, targs)
    prec = precision(preds, targs)

    f1 = fbeta_score(preds, targs)

    assert rec == 0.5
    assert prec == 0.5
    assert f1 == 0.5
