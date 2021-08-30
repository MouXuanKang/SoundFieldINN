# PDDO-PINN-Helmholtz
## Overview
Based on the physical information neural network framework (PINNs) to achieve neural network forecasting of sound field information in water.

From various aspects such as computational efficiency and computational accuracy, it effectively improves the inefficiency of 3D sound field computation.

The main technology of the project contains Peripheral Dynamics Operator (PDDO)

## How to use:

Dataset: run Dataset.py to get .pickle and .mat
```
Dataset.py
```

My model:
```
main.py
```
## Features:
- [x] Create dataset, include complete pressure.
- [ ] Load dataset, random sample dataset.Create Model. 
- [ ] train and test.

## Issues: 
Instead of feeding the network with [1,1]arrays, each batch is fed with [1,49]arrays.But incomplete due to a Type error.

TypeError: Failed to convert object of type <class 'sciann.functionals.variable.Variable'> to Tensor. Contents: <sciann.functionals.variable.Variable object at 0x7fd4252b18d0>. Consider casting elements to a supported type.

## References：
1. Haghighat E, Bekar A C, Madenci E, et al. A nonlocal physics-informed deep learning framework using the peridynamic differential operator[J]. Computer Methods in Applied Mechanics and Engineering, 2021, 385: 114012.
2. Madenci E, Barut A, Dorduncu M. Peridynamic differential operator for numerical analysis[M]. Springer International Publishing, 2019.
3. Madenci E, Barut A, Futch M. Peridynamic differential operator and its applications[J]. Computer Methods in Applied Mechanics and Engineering, 2016, 304: 408-451.

**Thanks to the very powerful framework [SciANN](https://github.com/sciann/sciann) built by Ehsan, I was able to do a little application work on this framework.**

For more details, check out [Ehsan's paper](https://arxiv.org/abs/2005.08803) and the [documentation](SciANN.com).

Ehsan created a [community](https://app.slack.com/client/T010WP0KD39/C010G71GXUJ) to discuss issues.
