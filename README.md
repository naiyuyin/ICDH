# Identifiable Causal Discovery under Heteroscedastic data (ICDH)

This repo is an implementation of the ICDH algorithm from following paper:\
[1] Naiyu Yin, Tian Gao, Yue Yu, and Qiang Ji. (2024). [Effective Causal Discovery under Heteroscedatic Noise Model](https://arxiv.org/abs/2312.12844). (AAAI 2024)

Please cite our paper with: 
```console
@inproceedings{yin2024effective,
  title={Effective Causal Discovery under Identifiable Heteroscedastic Noise Model},
  author={Yin, Naiyu and Gao, Tian and Yu, Yue and Ji, Qiang},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2024}
```


## Requirements for installation.

- python>=3.7
- numpy >=1.19.5
- scipy >= 1.10.1
- torch >= 1.10.2
- scikit-learn >= 1.3.2
- python-igraph
- time
- argparse

## Example script for generating the nonlinear (heteroscedastic noise) data
```console
$ git clone https://github.com/naiyuyin/ICDH.git 
$ cd ICDH
$ python data_generation.py --num_size 5 --s0 2 --sem mlp --graph_type ER --sample_size 1000 --data_type hetero
```
## Example script for running the algorithm.
After running the above script to generate nonlinear hetero data, one can try our ICDH algorithm by running the following example script:\
```console
$ python train.py --lamb1 0.01 --lamb2 0.01
```
where `lamb1` and `lamb2` are the coefficients for the L1 and L2 regularization. 
