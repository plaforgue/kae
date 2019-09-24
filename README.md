# Kernel Autoencoders

## Summary

This Python package contains code to use Kernel Autoencoders (KAEs). The main class is `KAE`. It is instantiated using information about the successive kernel functions, _i.e._ kernel types, hyperparameters, output matrices, regularizers. Then, it can be trained, using the `train` method, either on standard input data (for standard KAEs), or only based on Gram matrices (for the kernelized KAEs). The successive representations of a test input (standard or Gram) can finally be obtained through the `predict` method.

## Installation
To install the package, simply clone it, and then do:

  `$ pip install -e .`

To check that everything worked, the command

  `$ python -c 'import kae'`

should not return any error.


## Use
See the toy example available at `toy_example/toy_example.py`.
