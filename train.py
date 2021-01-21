from __future__ import division
from __future__ import print_function

from deeprobust.graph.data import Dataset, PrePtbDataset
import time
import argparse
import numpy as np
import scipy.sparse as ssp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import accuracy, normalize
from models import GCN, GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode',
                    action='store_true',
                    default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs',
                    type=int,
                    default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr',
                    type=float,
                    default=0.02,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay',
                    type=float,
                    default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden',
                    type=int,
                    default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout',
                    type=float,
                    default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--gcn-model',
                    type=str,
                    default='GCN',
                    choices=['GCN', 'GAT'],
                    help='select the encoding model to train GCN/GAT/JKNet')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')
parser.add_argument('--beta',
                    type=float,
                    default=0.05,
                    help='rate of pseudo attack sample')
parser.add_argument('--gamma',
                    type=float,
                    default=0.05,
                    help='rate of pseudo attack sample')
parser.add_argument('--ptb-rate',
                    type=str,
                    default='0.25',
                    help='perturbation parameters')
parser.add_argument('--a',
                    type=float,
                    default=10,
                    help='weight of bce loss')

args = parser.parse_args()
args.delta = args.beta + args.gamma + float(args.ptb_rate)
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# from deeprobust.graph.utils import *
data = Dataset(root='/tmp/', name=args.dataset, seed=15)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

# load pre-attacked graph
if args.ptb_rate != '0':
    perturbed_data = PrePtbDataset(root='/tmp/',
                                   name=args.dataset,
                                   attack_method='meta',
                                   ptb_rate=float(args.ptb_rate))
    adj_a = perturbed_data.adj
else:
    adj_a = adj

# print(idx_train, idx_val, idx_test)
args.nclass = int(labels.max() + 1)
node_num = int(adj_a.shape[0])
edge_num = int(adj_a.sum())
train_num = int(idx_train.shape[0])

# encoder and optimizer
if args.gcn_model == 'GCN':
    encoder = GCN(nfeat=features.shape[1],
                  nhid=args.hidden,
                  nclass=args.nclass,
                  dropout=0.5)
elif args.gcn_model == 'GAT':
    encoder = SpGAT(nfeat=features.shape[1],
                    nhid=8,
                    nclass=args.nclass,
                    dropout=0.5,
                    alpha=0.2,
                    nheads=8)
else:
    import warnings
    warnings.warn('{} is not supported for encoding yet. exiting...'.format(
        args.gcn_model))
    exit()
attacker = nn.Sequential(nn.Linear(args.nclass * 2 + 1, args.hidden),
                         nn.LeakyReLU(negative_slope=0.2),
                         nn.Dropout(0.5),
                         nn.BatchNorm1d(args.hidden),
                         nn.Linear(args.hidden, 1))
detector = nn.Sequential(nn.Linear(args.nclass * 2 + 1, args.hidden),
                         nn.LeakyReLU(negative_slope=0.2),
                         nn.Dropout(0.5),
                         nn.BatchNorm1d(args.hidden),
                         nn.Linear(args.hidden, 1))

optimizer_enc = optim.Adam(encoder.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
optimizer_att = optim.Adam(
    attacker.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay)
optimizer_det = optim.Adam(
    detector.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay)

adj_a = torch.FloatTensor((adj_a + ssp.eye(adj_a.shape[0])).todense())
adj = torch.FloatTensor((adj + ssp.eye(adj.shape[0])).todense())
features = torch.FloatTensor(features.todense())
labels = torch.LongTensor(labels)
onehot_labels = torch.eye(args.nclass)[labels]
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
onehot_labels = torch.eye(args.nclass)[labels].to(torch.long)

tmpdiag = torch.eye(node_num)  # .cuda() if args.cuda else torch.eye(node_num)
isedge_uv = torch.stack(torch.where((adj_a - tmpdiag) > 0.5), dim=0)
isedge_uv = isedge_uv[:, isedge_uv[0] < isedge_uv[1]]
isedge_idx = (isedge_uv[0] * node_num + isedge_uv[1]).to(torch.long)
noedge_uv = torch.stack(torch.where(adj_a < 0.5), dim=0)
noedge_uv = noedge_uv[:, noedge_uv[0] < noedge_uv[1]]
noedge_idx = (noedge_uv[0] * node_num + noedge_uv[1]).to(torch.long)
noedge_arange = np.arange(noedge_uv.shape[1])

if args.cuda:
    encoder.cuda()
    attacker.cuda()
    detector.cuda()
    features = features.cuda()
    adj_a = adj_a.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    onehot_labels = onehot_labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    onehot_labels = onehot_labels.cuda()
features = normalize(features, method='rowwise').detach()


def init_epoch():
    encoder.train()
    attacker.train()
    detector.train()
    optimizer_enc.zero_grad()
    optimizer_att.zero_grad()
    optimizer_det.zero_grad()


def train_gcn(epoch, denoising_mt, adj_a):
    # train gcn model

    init_epoch()

    new_adj = adj_a + denoising_mt

    if 'GAT' not in args.gcn_model:
        new_adj = normalize(new_adj, method='symmetric')
    embedding = encoder(features, new_adj)
    pred = F.log_softmax(embedding, dim=-1)

    acc_train = accuracy(pred[idx_train], labels[idx_train])
    loss_ce = F.nll_loss(pred[idx_train], labels[idx_train])

    loss_train = loss_ce
    loss_train.backward()
    optimizer_enc.step()

    if not args.fastmode:
        encoder.eval()
        embedding = encoder(features, new_adj)
        pred = F.softmax(embedding, dim=-1)
    acc_val = accuracy(pred[idx_val], labels[idx_val])
    acc_test = accuracy(pred[idx_test], labels[idx_test])

    print(
        "Epoch: {:04d}, train set loss({:.4f}) = cross_entropy({:.4f})  accuracy={:.4f}, val-acc={:.4f}, test-acc={:.4f}"
        .format(epoch, loss_train.item(), loss_ce.item(), acc_train.item(),
                acc_val.item(), acc_test.item()))
    return acc_val, acc_test


def get_candidate_uv():
    candidate_noedge_idx = np.random.choice(noedge_arange,
                                            size=isedge_uv.shape[1],
                                            replace=False)
    candidate_noedge_uv = noedge_uv[:, candidate_noedge_idx]
    candidate_uv = [
        isedge_uv, candidate_noedge_uv, candidate_noedge_uv[[1, 0]].flip(1),
        isedge_uv[[1, 0]].flip(1)
    ]
    candidate_uv = torch.cat(candidate_uv, dim=-1)

    candidate_idx = (candidate_uv[0] * node_num + candidate_uv[1]).to(
        torch.long)
    device = 'cuda' if args.cuda else 'cpu'
    return candidate_uv.to(device), candidate_idx.to(device)


def generate_s(edge_score, candidate_idx, delta_adj1d, attack_rate):
    attack_num = int(edge_num * attack_rate)
    threshold = edge_score.sort(descending=True)[0][attack_num].item()
    edge_score = torch.threshold(edge_score, threshold, 0)
    pseudo_label = (edge_score > 0).to(torch.float)
    s = torch.zeros(node_num * node_num)
    if args.cuda:
        s = s.cuda()
    s[candidate_idx] = delta_adj1d[candidate_idx] * edge_score
    return s.reshape([node_num, node_num]), pseudo_label


def train_att(epoch, candidate_uv, candidate_idx, adj_a):
    # train attack model for pseudo label for detection and denoising

    def cw_loss(outputs, onehot_labels):
        # one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        othermax, _ = torch.max((1 - onehot_labels) * outputs, dim=1)
        ground_truth = torch.masked_select(outputs, onehot_labels.bool())

        # origin implementation:
        # return torch.clamp(self._targeted*(j-i), min=-self.kappa)
        # but self._targeted is setted as 1.
        # print((j-i).min(),(j-i).max())
        # print(ground_truth,othermax)
        return (ground_truth - othermax).sum()
        return torch.clamp(ground_truth - othermax, min=-1).sum()

    init_epoch()
    delta_adj1d = (1 - adj_a * 2).reshape(-1)
    # generate feedin
    encoder.eval()
    embedding = encoder(features, normalize(adj_a)).detach()
    feedin = torch.cat([
        embedding[candidate_uv[0]], embedding[candidate_uv[1]],
        # adj_a.reshape(-1)[candidate_idx].unsqueeze(1)
        adj_a[(candidate_uv[0], candidate_uv[1])].unsqueeze(1)
    ],
        dim=-1)
    # generate edge_score and attack sample
    edge_score = attacker(feedin).squeeze()
    edge_score = (edge_score + edge_score.flip(0)) / 2
    attack_sample, pseudo_label = generate_s(F.sigmoid(edge_score),
                                             candidate_idx, delta_adj1d,
                                             args.beta)
    # evaluate attack samples and update attacker
    embedding = encoder(features, normalize(adj_a + attack_sample))
    pred = F.softmax(embedding, dim=-1)
    loss_att = cw_loss(pred[idx_train], onehot_labels[idx_train])
    loss_att.backward()
    optimizer_att.step()
    print("Epoch: {:04d}, attack cw-loss = {:.4f}".format(
        epoch, loss_att.item()))
    # generate final attack sample
    if not args.fastmode:
        attacker.eval()
        edge_score = attacker(feedin).squeeze()
        edge_score = (edge_score + edge_score.flip(0)) / 2
        attack_sample, pseudo_label = generate_s(F.sigmoid(edge_score),
                                                 candidate_idx, delta_adj1d,
                                                 args.beta)

    adj_aa = adj_a + attack_sample
    return adj_aa.detach(), pseudo_label.detach(), attack_sample.detach()


def train_det(epoch, candidate_uv, candidate_idx, adj_aa, pseudo_label, attack_sample):
    # train for attack sample detection and denoising
    init_epoch()

    def cw_loss(outputs, onehot_labels):
        # one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        othermax, _ = torch.max((1 - onehot_labels) * outputs, dim=1)
        ground_truth = torch.masked_select(outputs, onehot_labels.bool())

        return (othermax - ground_truth).sum()
        return torch.clamp(othermax - ground_truth, min=-1).sum()

    # generate feedin
    encoder.eval()
    embedding = encoder(features, normalize(adj_aa)).detach()
    feedin = torch.cat([
        embedding[candidate_uv[0]], embedding[candidate_uv[1]],
        adj_aa[(candidate_uv[0], candidate_uv[1])].unsqueeze(1)
    ],
        dim=-1)
    # generate edge_score and attack sample
    edge_score = detector(feedin).squeeze()
    edge_score = (edge_score + edge_score.flip(0)) / 2

    # evaluate detection results and update detector
    # classification: BCE loss based on pseudo_label
    loss_bce = F.binary_cross_entropy_with_logits(edge_score,
                                                  target=pseudo_label,
                                                  reduction='mean')

    edge_score = F.sigmoid(edge_score)
    delta_adj1d = (1 - 2 * adj_aa).reshape(-1)
    denoise_sample, _ = generate_s(edge_score, candidate_idx, delta_adj1d,
                                   args.delta)
    # denoise_sample = denoise_sample.reshape(-1)
    tmpidx = torch.where(attack_sample != 0)
    denoise_sample[tmpidx] = -attack_sample[tmpidx]
    denoise_sample = denoise_sample.reshape([node_num, node_num])

    embedding = encoder(features, normalize(adj_aa + denoise_sample))
    pred = F.softmax(embedding, dim=-1)
    loss_gnn = cw_loss(pred[idx_train], onehot_labels[idx_train])
    # get final loss and update
    loss_det = args.a * loss_bce + loss_gnn
    # loss_det = loss_gnn
    loss_det.backward()
    optimizer_det.step()
    print(
        "Epoch: {:04d}, Detector, loss({:.4f}) = loss_bce({:.4f}) + loss_cw({:.4f})"
        .format(epoch, loss_det.item(), loss_bce, loss_gnn.item()))
    if not args.fastmode:
        detector.eval()
        edge_score = detector(feedin).squeeze()
        edge_score = (edge_score + edge_score.flip(0)) / 2
        edge_score = F.sigmoid(edge_score)
        denoise_sample, _ = generate_s(edge_score, candidate_idx, delta_adj1d,
                                       args.delta)
        tmpidx = torch.where(attack_sample != 0)
        denoise_sample[tmpidx] = -attack_sample[tmpidx]
        denoise_sample = denoise_sample.reshape([node_num, node_num])
    return denoise_sample.detach()


def alternately_train(epoch):
    candidate_uv, candidate_idx = get_candidate_uv()

    adj_aa, pseudo_label, attack_sample = train_att(
        epoch, candidate_uv, candidate_idx, adj_a)

    denoising_mt = train_det(epoch, candidate_uv, candidate_idx, adj_aa,
                             pseudo_label, attack_sample)

    acc_val, acc_test = train_gcn(epoch, denoising_mt, adj_aa)
    return acc_val, acc_test


begintime = time.time()

bestvalacc = 0
nowtestacc = 0
for epoch in range(args.epochs):
    acc_val, acc_test = alternately_train(epoch)
    if acc_val > bestvalacc:
        bestvalacc = acc_val
        nowtestacc = acc_test
    print(
        f"Epoch: {epoch:04d}, best valid acc = {bestvalacc:.4f}, now test acc = {nowtestacc:.4f}"
    )
