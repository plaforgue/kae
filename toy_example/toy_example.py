import torch
from kae import KAE
import matplotlib.pyplot as plt

torch.manual_seed(1)


# Global parameters
n = 100                           # number of samples
d_in, d_hid = 10, 5               # layers dimensions
layer_dims = [d_in, d_hid, d_in]
d_tot = d_in + d_hid
L = len(layer_dims) - 1

# Data
X_tr = torch.rand(n, d_in, dtype=torch.float64)  # train data
X_te = torch.rand(n, d_in, dtype=torch.float64)  # test data
G_in_tr = torch.mm(X_tr, torch.t(X_tr))          # train Gram matrix
G_in_te = torch.mm(X_te, torch.t(X_tr))          # test Gram matrix
G_in_te_own = torch.mm(X_te, torch.t(X_te))      # own test Gram matrix

# KAE parameters
gammas = torch.rand(L, dtype=torch.float64)
lambdas = torch.rand(L, dtype=torch.float64)
kernels = ['precomputed'] + ['gaussian'] * (L - 1)
kparams = ['useless'] + list(gammas[1:])
outmats = [1. * torch.eye(layer_dims[j], dtype=torch.float64)
           for j in range(1, L + 1)]
o_outmats = [1. * torch.eye(layer_dims[j], dtype=torch.float64)
             for j in range(1, L)] + ['identity']

# Initial coefficients
Phi_init = torch.rand(n * d_tot, dtype=torch.float64)
Phi_init_int = Phi_init[:n * d_hid]


# Instantiate standard KAE and train on X_tr (G_in to avoid computations)
kae1 = KAE(kernels, kparams, outmats, lambdas)
kae1.set_Phi(Phi_init)
kae1.fit(X=X_tr, Y=X_tr, G_in=G_in_tr, method='md',
         n_loop=40, n_epoch=10, solver='sgd', momentum=0.9, lr=0.001)


# Instantiate kernelized KAE and train on G_in_tr
kae2 = KAE(kernels, kparams, o_outmats, lambdas)
kae2.set_Phi(Phi_init_int)
kae2.set_NL(G_in=G_in_tr, G_out=G_in_tr)
kae2.fit(G_in=G_in_tr, G_out=G_in_tr, method='omd',
         n_loop=40, n_epoch=10, solver='sgd', momentum=0.9, lr=0.001)


# As the kernel has been chosen linear, the training must be (almost) the same
plt.figure()
plt.plot(kae1.losses[:50], label='standard kae')
plt.plot(kae2.losses[:50], label='kernelized kae')
plt.xlabel("Epochs")
plt.ylabel("Train loss")
plt.legend()
plt.show()


# Recover encoding on test data using the .predict method
kae1_code = kae1.predict(X_te=X_te, G_in_te=G_in_te)[1]
kae2_code = kae2.predict(G_in_te=G_in_te)[1]
norm = torch.norm(kae1_code)
print("Norm difference between codes: %.2e" % norm)
