import scipy.sparse as ssp
import numpy as np
import torch


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx, method='symmetric'):
    """method = {rowwise, colwise, symmetric}"""
    if method == 'rowwise':
        return mx / (mx.sum(-1, keepdims=True)+1e-8)
    elif method == 'colwise':
        return mx / (mx.sum(-2, keepdims=True)+1e-8)
    elif method == 'symmetric':
        rowsum = (mx.sum(-1, keepdims=True) + 1e-8)**(0.5)
        colsum = (mx.sum(-2, keepdims=True) + 1e-8)**(0.5)
        return mx / rowsum / colsum
