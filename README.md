# AGD

## Prerequisites

please install following package before running code

```
torch == 1.9.0+cu111
torch-geometric == 1.7.2
deeprobust == 0.2.2
scipy == 1.7.0
```

for installing torch_geometric, please refer to https://github.com/pyg-team/pytorch_geometric

for installing deeprobust, please refer to https://github.com/DSE-MSU/DeepRobust

## Running code

```
python train --dataset {name_of_dataset} --ptb_rate {rate_of_Mettack} --gcn_model {GCN or GAT} 
```
where `name_of_dataset` can be `['cora','citeseer','pubmed','cora_ml','polblogs','acm']`, and `rate_of_Mettack` can be `[0,0.05,0.1,0.15,0.2,0.25]`.

If you want to test the code on the non-attacked graph, just keep the `rate_of_Mettack` as `0`.