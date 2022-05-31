import torch.nn.functional as F
import scipy.sparse as ssp
import numpy as np
import torch
from models import AGD
from deeprobust.graph.data import Dataset, PrePtbDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--fastmode', type=bool, default=True)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--cuda_id', type=int, default=0)
parser.add_argument('--gcn_model', type=str, default='GCN')
parser.add_argument('--dataset', type=str, default='pubmed',choices=['cora','citeseer','pubmed','cora_ml','polblogs','acm'])
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--ptb_rate', type=float, default=0.2,help='the rate of Mettack in the paper'))
parser.add_argument('--attack_rate', type=float, default=0.2,help='beta in the paper')
parser.add_argument('--denoise_rate', type=float, default=0.02,help='gamma in the paper')
parser.add_argument('--lmda', type=float, default=0.1)

args = parser.parse_args()

args.device = device = torch.device(
    f'cuda:{args.cuda_id:d}' if torch.cuda.is_available() else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

data = Dataset(root='./datasets/',
               name=args.dataset, seed=15, setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

if args.ptb_rate != 0:
    adj = ssp.load_npz(
        f'./datasets/{args.dataset}_meta_adj_{args.ptb_rate:g}.npz')

features = torch.Tensor(features.todense()).to(args.device)
adj = adj.tocoo()
edge_index = torch.LongTensor([adj.row, adj.col]).to(args.device)
edge_values = torch.Tensor(adj.data).to(args.device)
labels = torch.LongTensor(labels).to(args.device)
idx_train = torch.LongTensor(idx_train).to(args.device)
idx_val = torch.LongTensor(idx_val).to(args.device)
idx_test = torch.LongTensor(idx_test).to(args.device)

args.num_nodes = features.shape[0]
args.num_features = features.shape[1]
args.num_classes = int(max(labels) + 1)
args.num_hiddens = 16
args.num_heads = 8

model: AGD = AGD(args).to(device)

best_gval, best_gtest = 0, 0
best_eval, best_etest = 0, 0

for args.epoch in range(1, args.epochs + 1):
    gval, gtest, eval, etest = model.train_all(
        features, edge_index, edge_values, labels, idx_train, idx_val, idx_test)
    if gval > best_gval:
        best_gval = gval
        best_gtest = gtest

    if eval > best_eval:
        best_eval = eval
        best_etest = etest

print(f"This is the result of {args.dataset}")
print(f"{args.gcn_model} test accuracy: {best_gtest:.4f}")
print(f"Encoder test accuracy: {best_etest:.4f}")