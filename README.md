# BayesianModelSelection
This repository contains the python code that was presented for the following paper, which has been submitted to IFAC for possible publication.

[1] Adachi, M., Kuhn, Y., Horstmann, B., Osborne, M. A., Howey, D. A. 
Bayesian Model Selection of Lithium-Ion Battery Models via Bayesian Quadrature, arXiv, 2022[https://arxiv.org/abs/2210.17299]

This work is based on the following paper, the link to its repository is [here](https://github.com/ma921/BASQ)

[2] Adachi, M., Hayakawa, S., Jørgensen, M., Oberhauser, H., Osborne, M. A.,
Fast Bayesian Inference with Batch Bayesian Quadrature via Kernel Recombination, NeurIPS 35, 2022 [link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/697200c9d1710c2799720b660abd11bb-Abstract-Conference.html)

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

## Getting started
Open "ECM_model_selection.ipynb".
This will give you a step-by-step introduction.

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
Also please consider to cite this work as well.
```
@article{adachi2022fast,
  title={Fast {B}ayesian Inference with Batch {B}ayesian Quadrature via Kernel Recombination},
  author={Adachi, Masaki and Hayakawa, Satoshi and J{\o}rgensen, Martin and Oberhauser, Harald and Osborne, Michael A},
  journal={Advances in neural information processing systems (NeurIPS)},
  volume={35},
  year={2022},
}
```
