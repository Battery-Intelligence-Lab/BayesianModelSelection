# BayesianModelSelection
This repository contains the python code that was presented for the following paper.

[1] Adachi, M., Kuhn, Y., Horstmann, B., Osborne, M. A., Howey, D. A. 
Bayesian Model Selection of Lithium-Ion Battery Models via Bayesian Quadrature, arXiv, 2022[https://arxiv.org/abs/2210.17299]

This work has been submitted to IFAC for possible publication.
![plot](./overview.png)

## Features
- fast Bayesian inference via Bayesian quadrature
- Simultaneous inference of Bayesian model evidence and posterior
- GPU acceleration
- Canonical equivalent circuit model (ECM)
- Statistical analysis computation of the ECM

## Requirements
- PyTorch
- GPyTorch
- BoTorch
- functorch

## Cite as

Please cite this work as
```
@misc{adachi2022bayesian,
  title={Bayesian Model Selection of Lithium-Ion Battery Models via Bayesian Quadrature},
  author={Adachi, Masaki and Kuhn, Yannick and Horstmann, Birger and Osborne, Michael A. and Howey, David A.},
  publisher = {arXiv},
  year={2022}
  doi = {10.48550/ARXIV.2210.17299},
}
```
