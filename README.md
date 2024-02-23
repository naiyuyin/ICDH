# Identifiable Causal Discovery under Heteroscedastic data (ICDH)

This repo is an implementation of the ICDH algorithm from the AAAI24 paper "Effective Causal Discovery under Heteroscedatic Noise Model". 

Our paper can be found   



## Requirements for installation.

> numpy \
> torch
> 

## Example script for generating the nonlinear (heteroscedastic noise) data
> python data_generation.py --num_size 5 --s0 2 --sem mlp --graph_type ER --sample_size 1000 --data_type hetero

## Example script for running the algorithm.


