please install following package before running code 

pytorch == 1.5.0

deeprobust == 0.1.1


if you want to test the performance on PolBlogs with 5% attacking edges, please run the following command:

**python train.py --lr 0.02 --gcn-model GCN --dataset polblogs --a 10 --beta 0.1 --gamma 0.2  --ptb-rate 0.05 --dataset polblogs**

where '--a' is the weight of Loss_{bce}, '--beta' is the ratio of auxiliary attacks, '--gamma' is the ratio of background noise and '--gcn-model' can be GCN or GAT. 

'--ptb-rate' is the ratio of attacking edges, and if you want to test the performance on real-world datasets, use '--ptb-rate 0'.
