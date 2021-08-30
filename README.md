# Sound field information neural network
Sound field information predict with PINNs framwork, in this project-test i try to use peridynamic differential operator...

## Table of Contents

1. [Overview](#Overview)
2. [Usage](#Usage)
3. [Features](#Features)
4. [Current Issues](#Current-Issues)
5. [Background](#Background)
6. [References](#References)

## Overview
~~project undone, hopeing i can finish it...~~  
My project scheduled in "Features".If i finish this project, i'll put features in here.  
I'll put the questions I feel stuck on me in "Current-Issues" and welcome your advice.  
I explain my project simplely in "Background".  

## Usage:

Dataset: run Dataset.py to get .pickle and .mat
```
$ DataSet.py
```

My model:
```
$ main.py
```
## Features:
- [x] Create dataset, include complete pressure.
- [ ] Load dataset, random sample dataset.Create Model. 
- [ ] train and test.

## Current Issues: 
It looks like I got the type input wrong, unit and shape are two kind Arg.

*TypeError: Failed to convert object of type <class 'sciann.functionals.variable.Variable'> to Tensor. Contents: <sciann.functionals.variable.Variable object at* *0x7fd4252b18d0>. Consider casting elements to a supported type.*   
~~Very stupid mistake,and i'm learning keras to correct it...~~

## Background:  
I get real-pressure and image-pressure by other method.And choose Family with size [7,7].  
>*Governing equations is Inhomogeneous Helmholtz equation.* 
>Figure below is transmission of pressure with range and depth.  
>Ofcourse i split complex pressure in real-part and image-part.So there is two dataset.
>Actually,every application has it's PDEs,i think we should care how to make model work...

<p align="center">
  <img src="./figures/fig6.png" width="306" height="205">
  <img src="./figures/fig5.png" width="306" height="205">
</p>

PINNs results, with *split-stepping method*(or i can't start trainning, LOSS always high): 
<p align="center">
  <img src="./figures/fig5-1.png" width="606" height="205">
</p>

Figure right is error-pointwise,seems to be working with split-stepping.But part of near-field place is not work.So may PDDO could solve that?
>welcome to discuse abous the how to make model working.It seem that we can's use sciANN direct solve our engerneering problem.
## References：
1. Haghighat E, Bekar A C, Madenci E, et al. A nonlocal physics-informed deep learning framework using the peridynamic differential operator[J]. Computer Methods in Applied Mechanics and Engineering, 2021, 385: 114012.
2. Madenci E, Barut A, Dorduncu M. Peridynamic differential operator for numerical analysis[M]. Springer International Publishing, 2019.
3. Madenci E, Barut A, Futch M. Peridynamic differential operator and its applications[J]. Computer Methods in Applied Mechanics and Engineering, 2016, 304: 408-451.

**Thanks to the very powerful framework [SciANN](https://github.com/sciann/sciann) built by Ehsan, I was able to do a little application work on this framework.**

For more details, check out [Ehsan's paper](https://arxiv.org/abs/2005.08803) and the [documentation](SciANN.com).

Ehsan created a [community](https://app.slack.com/client/T010WP0KD39/C010G71GXUJ) to discuss issues.
