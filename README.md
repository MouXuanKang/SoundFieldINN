# PDDO-PINN-Helmholtz
Based on the physical information neural network framework (PINNs) to achieve neural network forecasting of sound field information in water.

From various aspects such as computational efficiency and computational accuracy, it effectively improves the inefficiency of 3D sound field computation.

The main technology of the project contains Peripheral Dynamics Operator (PDDO)

# How to use:

Dataset: run Dataset.py to get .pickle and .mat

My model:
run main.py

# Progress: 
Instead of feeding the network with [1,1]arrays, each batch is fed with [1,49]arrays.But incomplete due to a Type error.

TypeError: Failed to convert object of type <class 'sciann.functionals.variable.Variable'> to Tensor. Contents: <sciann.functionals.variable.Variable object at 0x7fd4252b18d0>. Consider casting elements to a supported type.

