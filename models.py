import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops, add_self_loops
from torch_scatter import scatter_add
import torch_sparse
from torch_geometric.utils.num_nodes import maybe_num_nodes

from gcn_conv import GCNConv
from gat_conv import GATConv

import utils
import numpy as np


class Sampler:

    def __init__(self, args, gcn_model):
        self.gcn_model = gcn_model
        self.args = args

    def sample(self, x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test):

        embedding, _ = self.gcn_model(x, edge_index, edge_weight)
        embedding = F.softmax(embedding, dim=1)

        n = x.shape[0]
        m = edge_index.shape[-1]
        size = int(np.sqrt(m // 2))
        entropy = torch.distributions.Categorical(probs=embedding).entropy()
        com_ent = torch.clamp_min(np.log(self.args.num_classes) - entropy, 0)

        centroid = torch.multinomial(com_ent, size, replacement=False)
        borderline = torch.multinomial(entropy, size, replacement=False)

        non_exist_edge_index = torch.cartesian_prod(centroid, borderline).t()
        non_exist_edge_index = torch.cat(
            [non_exist_edge_index, torch.flip(non_exist_edge_index, dims=[0])], dim=1)

        sampled_edge_index = torch.cat(
            [edge_index, non_exist_edge_index], dim=1)
        sampled_edge_weight = torch.cat(
            [edge_weight, torch.zeros(non_exist_edge_index.shape[1], dtype=torch.float, device=self.args.device)])

        sampled_edge_index, sampled_edge_weight = torch_sparse.coalesce(
            sampled_edge_index, sampled_edge_weight, n, n, 'add')

        mask = sampled_edge_index[0] < sampled_edge_index[1]
        sampled_edge_index = sampled_edge_index[:, mask]
        sampled_edge_weight = sampled_edge_weight[mask]
        sampled_edge_index = torch.cat(
            [sampled_edge_index, torch.flip(sampled_edge_index, dims=[0,1])], dim=1)
        sampled_edge_weight = torch.cat([sampled_edge_weight,sampled_edge_weight.flip(dims=[0])],dim=0)
        assert torch.allclose(sampled_edge_index[0],sampled_edge_index[1].flip(dims=[0]))
        return sampled_edge_index, sampled_edge_weight


class GCN(nn.Module):
    # inspired by torch_geometric: https://github.com/rusty1s/pytorch_geometric

    def __init__(self, args):
        super(GCN, self).__init__()

        self.num_nodes = args.num_nodes

        self.conv1 = GCNConv(args.num_features, 16)
        self.conv2 = GCNConv(16, args.num_classes)

        self.device = args.device
        self.args = args

    def forward(self, x, edge_index, edge_weight):
        edge_index, edge_weight = self.gcn_norm(
            edge_index, edge_weight, None)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        hidden = x
        x = F.dropout(x, 0.5, training=self.training)
        return self.conv2(x, edge_index, edge_weight), hidden

    def gcn_norm(self, edge_index, edge_weight=None, num_nodes=None, improved=False, dtype=None):
        fill_value = 2. if improved else 1.

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        edge_index, tmp_edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GAT(nn.Module):
    # inspired by torch_geometric: https://github.com/rusty1s/pytorch_geometric

    def __init__(self, args):
        super(GAT, self).__init__()

        self.num_nodes = args.num_nodes
        self.num_hiddens = 8

        self.attention = GATConv(
            args.num_features, self.num_hiddens, args.num_heads)
        self.out_att = GATConv(
            self.num_hiddens * args.num_heads, args.num_classes)

        self.device = args.device
        self.args = args

    def forward(self, x, edge_index, edge_weight):
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, 1, self.num_nodes)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.attention(x, edge_index, edge_weight)
        x = F.elu(x)
        hidden = x
        x = F.dropout(x, 0.5, training=self.training)
        return self.out_att(x, edge_index, edge_weight), hidden


class SurrogateModel(nn.Module):

    def __init__(self, args):

        super(SurrogateModel, self).__init__()

        if args.gcn_model == 'GCN':
            self.model = GCN(args)
        else:
            self.model = GAT(args)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.device = args.device
        self.args = args

    def forward(self, x, edge_index, edge_weight):
        return self.model.forward(x, edge_index, edge_weight)

    def fit(self, x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test):

        # Training
        self.optimizer.zero_grad()
        self.model.train()
        embedding, _ = self.model.forward(x, edge_index, edge_weight)
        pred = F.log_softmax(embedding, dim=1)

        loss = F.nll_loss(pred[idx_train], labels[idx_train])
        loss.backward()
        self.optimizer.step()

        # Evaluation
        self.model.eval()
        embedding, _ = self.model.forward(x, edge_index, edge_weight)
        acc_train = utils.accuracy(
            embedding[idx_train], labels[idx_train]).item()
        acc_val = utils.accuracy(embedding[idx_val], labels[idx_val]).item()
        acc_test = utils.accuracy(embedding[idx_test], labels[idx_test]).item()
        # print(
        #     f"Epoch {self.args.epoch:d}, loss = {loss.item():.4f}, acc_train = {acc_train:.4f}, acc_val = {acc_val:.4f}, acc_test = {acc_test:.4f}")
        return acc_train, acc_val, acc_test


class AGD(nn.Module):

    def __init__(self, args):
        super(AGD, self).__init__()

        self.gcn_model = SurrogateModel(args)
        self.sampler = Sampler(args, self.gcn_model)
        self.enc_model = nn.Sequential(
            nn.Linear(args.num_features, args.num_hiddens),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            nn.Linear(args.num_hiddens, args.num_classes)
        )

        if args.gcn_model == "GCN":
            scorer_input_channels = args.num_hiddens * 2 + 1
        elif args.gcn_model == "GAT":
            scorer_input_channels = 2 * 8 * 8 + 1
        self.scorer = nn.Sequential(
            nn.Linear(scorer_input_channels, args.num_hiddens),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(args.num_hiddens),
            nn.Dropout(),
            nn.Linear(args.num_hiddens, 1)
        )
        self.optim_gcn = torch.optim.Adam(
            self.gcn_model.parameters(), lr=args.lr if args.gcn_model == "GCN" else 0.005,
            weight_decay=args.weight_decay)
        self.optim_enc = torch.optim.Adam(
            self.enc_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.optim_sco = torch.optim.Adam(
            self.scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.attack_rate = args.attack_rate
        self.denoise_rate = args.denoise_rate

        self.device = args.device
        self.args = args

        self.original_modularity = None

    def symmetrization(self, edge_index, edge_weight):
        # edge_index = torch.cat(
        #     [edge_index, torch.flip(edge_index, dims=[0])], dim=1)
        # edge_weight = torch.cat([edge_weight, edge_weight])
        # n = self.args.num_nodes
        # return torch_sparse.coalesce(edge_index, edge_weight, n, n, 'mean')
        return edge_index, (edge_weight + edge_weight.flip(dims=[0]))/2

    def attack(self, x, edge_index, edge_weight):
        _, gcn_output = self.gcn_model(x, edge_index, edge_weight)
        gcn_output = gcn_output.detach()
        enc_output = self.enc_model(x).detach()

        edge_score: torch.Tensor = self.scorer(torch.cat(
            [
                gcn_output[edge_index[0]],
                gcn_output[edge_index[1]],
                # enc_output[edge_index[0]],
                # enc_output[edge_index[1]],
                edge_weight.unsqueeze(-1)
            ], dim=-1
        )).squeeze()
        edge_index, edge_score = self.symmetrization(
            edge_index, edge_score)

        quantile = torch.quantile(edge_score, 1 - self.attack_rate).item()
        # print(f"attack threshold: {quantile:.4f}")
        masked_edge_score = torch.threshold(edge_score, quantile, value=-1e9)
        # print(f"threshold: {quantile:.4f}")

        attack_values = (1 - 2 * edge_weight) * \
            torch.sigmoid(masked_edge_score)

        lower_bound = torch.quantile(edge_score, self.denoise_rate).item()
        pseudo_labels = ((edge_score > quantile) | (
            edge_score < lower_bound)).to(torch.float)

        return attack_values, pseudo_labels

    def denoise(self, x, edge_index, edge_weight, with_attack):
        rate = self.denoise_rate if not with_attack else self.denoise_rate + self.attack_rate
        _, gcn_output = self.gcn_model(x, edge_index, edge_weight)
        gcn_output = gcn_output.detach()
        enc_output = self.enc_model(x).detach()

        edge_score = -self.scorer(torch.cat(
            [
                gcn_output[edge_index[0]],
                gcn_output[edge_index[1]],
                # enc_output[edge_index[0]],
                # enc_output[edge_index[1]],
                edge_weight.unsqueeze(-1)
            ], dim=-1
        )).squeeze()
        edge_index, edge_score = self.symmetrization(edge_index, edge_score)
        quantile = torch.quantile(edge_score, 1 - rate).item()
        # if (not with_attack) and self.scorer.training:
        #     # quantile = quantile * 0
        #     quantile = min(quantile, 0)
        # print(f"denoise threshold: {quantile:.4f}")
        masked_edge_score = torch.threshold(edge_score, quantile, value=-1e9)

        denoise_values = (1 - 2 * edge_weight) * \
            torch.sigmoid(masked_edge_score)

        return denoise_values, edge_score

    def sample(self, x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test):
        return self.sampler.sample(x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test)

    def train_gcn(self, x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test, with_denoise=False):
        if with_denoise:
            self.scorer.eval()
            denoise_values, _ = self.denoise(
                x, edge_index, edge_weight, with_attack=False)

            edge_weight += denoise_values.detach()

        # Training
        self.zero_grad()
        self.gcn_model.train()
        embedding, _ = self.gcn_model.forward(x, edge_index, edge_weight)
        pred = F.log_softmax(embedding, dim=1)

        loss = F.nll_loss(
            pred[idx_train], labels[idx_train])
        loss.backward()
        self.optim_gcn.step()

        # Evaluation
        acc_train, acc_val, acc_test = self.test_gcn(x, edge_index, edge_weight, labels, idx_train, idx_val,
                                                     idx_test)
        # print(
        #     f"Epoch: {self.args.epoch:d}, loss = {loss.item():.4f}, acc_train = {acc_train:.4f}, acc_val = {acc_val:.4f}, acc_test = {acc_test:.4f}")
        return acc_val, acc_test

    def train_enc(self, x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test, with_denoise=True):
        # Training
        self.zero_grad()
        self.enc_model.train()
        embedding = self.enc_model.forward(x)
        pred = F.log_softmax(embedding, dim=1)

        if with_denoise:
            self.scorer.eval()
            denoise_values, _ = self.denoise(
                x, edge_index, edge_weight, with_attack=False)
            # print('(denoise_values.abs()>0).sum()=',(denoise_values.abs()>0).sum())
            edge_weight = edge_weight + denoise_values

        gcn_embedding, _ = self.gcn_model.forward(
            x, edge_index, edge_weight)
        gcn_embedding = gcn_embedding.detach()

        loss = F.nll_loss(pred[idx_train], labels[idx_train]) + F.kl_div(pred,
                                                                         F.softmax(
                                                                             gcn_embedding, dim=1),
                                                                         log_target=False,
                                                                         reduction='none').sum(1).mean(0)

        loss.backward()
        self.optim_enc.step()

        # Evaluation
        self.enc_model.eval()
        embedding = self.enc_model.forward(x)
        acc_train = utils.accuracy(embedding[idx_train], labels[idx_train])
        acc_val = utils.accuracy(embedding[idx_val], labels[idx_val])
        acc_test = utils.accuracy(embedding[idx_test], labels[idx_test])
        # print(
        #     f"Epoch: {self.args.epoch:d}, loss = {loss.item():.4f}, acc_train = {acc_train.item():.4f}, acc_val = {acc_val.item():.4f}, acc_test = {acc_test.item():.4f}")
        return acc_val, acc_test

    def train_att(self, x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test):
        # Training
        self.zero_grad()
        self.scorer.train()
        self.gcn_model.eval()
        self.enc_model.eval()
        attack_values, _ = self.attack(
            x, edge_index, edge_weight)

        def cw_loss(output, target):
            output = output[idx_train]
            target = target[idx_train]
            other_max = torch.max(
                output * (1 - F.one_hot(target)), dim=1).values
            ground_truth = output[torch.arange(
                output.shape[0], device=self.device), target]
            return (ground_truth - other_max).mean()
        attacked_embedding, _ = self.gcn_model(
            x, edge_index, edge_weight + attack_values)
        loss = cw_loss(F.softmax(attacked_embedding, dim=1), labels)
        loss.backward()
        self.optim_sco.step()


    def train_det(self, x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test):
        self.zero_grad()
        self.scorer.train()
        self.gcn_model.eval()
        self.enc_model.eval()

        attack_values, pseudo_labels = self.attack(
            x, edge_index, edge_weight)

        attack_values = torch.sign(attack_values).detach()
        _, edge_score = self.denoise(
            x, edge_index, edge_weight + attack_values, with_attack=True)
        denoise_values, _ = self.denoise(
            x, edge_index, edge_weight, with_attack=False)
        denoised_embedding, _ = self.gcn_model(
            x, edge_index, edge_weight + denoise_values)

        def cw_loss(output, target):
            output = output[idx_train]
            target = target[idx_train]
            other_max = torch.max(
                output * (1 - F.one_hot(target)), dim=1).values
            ground_truth = output[torch.arange(
                output.shape[0], device=self.device), target]
            return (other_max - ground_truth).mean()

        loss_bce = F.binary_cross_entropy_with_logits(
            edge_score, pseudo_labels)
        loss_cw = cw_loss(denoised_embedding, labels)
        # print(
        #     f"loss_bce = {loss_bce.item():.4f},loss_cw = {loss_cw.item():.4f}")
        loss = self.args.lmda * loss_bce + (1 - self.args.lmda) * loss_cw
        loss.backward()
        self.optim_sco.step()

    def train_secondary(self, x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test):
        edge_index, edge_weight = self.sampler.sample(
            x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test)
        gval, gtest = self.train_gcn(x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test,
                                     with_denoise=True)
        eval, etest = self.train_enc(
            x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test)
        return gval, gtest, eval, etest

    def train_all(self, x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test):
        edge_index, edge_weight = self.sampler.sample(
            x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test)
        self.train_att(x, edge_index, edge_weight, labels,
                        idx_train, idx_val, idx_test)
        self.train_det(x, edge_index, edge_weight, labels,
                        idx_train, idx_val, idx_test)
        gval, gtest = self.train_gcn(x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test,
                                        with_denoise=True)
        eval, etest = self.train_enc(
            x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test)
        return gval, gtest, eval, etest

    def test_gcn(self, x, edge_index, edge_weight, labels, idx_train, idx_val, idx_test):

        self.gcn_model.eval()
        embedding, _ = self.gcn_model.forward(x, edge_index, edge_weight)
        acc_train = utils.accuracy(embedding[idx_train], labels[idx_train])
        acc_val = utils.accuracy(embedding[idx_val], labels[idx_val])
        acc_test = utils.accuracy(embedding[idx_test], labels[idx_test])
        print(
            f"Epoch: {self.args.epoch:d}, GCN model, train acc = {acc_train:.4f}, val_acc = {acc_val:.4f}, test acc = {acc_test:.4f}")
        return acc_train.item(), acc_val.item(), acc_test.item()
