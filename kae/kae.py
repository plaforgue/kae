import time
import torch
import torch.optim as optim


###############################################################################

#                        Dictionaries functions

###############################################################################


def mk_ker_dic(kernels, kparams, outmats, lambdas):
    """Create the global kernel dictionary for the L layers

    Parameters
    ----------
    kernels: list of L str
             Successive kernels. Ex: "gaussian", "polynomial", "precomputed"

    kparams: list of L float/tuple/str
             Parameters associated to kernels

    outmats: list of L torch.Tensor of shape (n_features_l, n_features_l)/str
             Output matrices. "identity" if infinite dimensional output.
             Identity matrices are not stored for efficiency

    lambdas: list of L float
             Regularization parameters

    Returns
    -------
    ker_dic: dict
             Kernels and their parameters
             keys: {1, ..., L, 'dim_tot'}
             values: dict with kernel info / sum of intermediate spaces dims
    """
    # Create individual layer dico from 1 kernel, 1 param, 1 outmat, 1 lambda
    def mk_k_dic(kernel, kparam, outmat, lambda_):
        k_dic = {'kernel': kernel,
                 'kparam': kparam,
                 'lambda': lambda_}
        if isinstance(outmat, torch.Tensor):
            k_dic['outdim'] = outmat.shape[0]
            # If outmat = torch.eye(d), no need to store it
            if torch.norm(outmat - torch.eye(outmat.shape[0],
                                             dtype=torch.float64)) > 1e-10:
                k_dic['outmat'] = outmat.clone()
        elif outmat == 'identity':
            k_dic['outmat'] = 'identity'
            k_dic['outdim'] = 'infty'
        return k_dic

    # Return the dico of individual layer dicos + 'dim_tot'
    L = len(kernels)
    ker_dic = {'dim_tot': 0}

    for l in range(L):
        ker_dic[l + 1] = mk_k_dic(kernels[l], kparams[l], outmats[l],
                                  lambdas[l])
        # Count total dimension (exception with 'infty')
        try:
            ker_dic['dim_tot'] += ker_dic[l + 1]['outdim']
        except TypeError:
            pass

    return ker_dic


def mk_Phi_dic(Phi_ravel, ker_dic):
    """Structure Phi_ravel into layers

    Parameters
    ----------
    Phi_ravel: torch.Tensor of shape (n_samples * dim_tot, )
               All coefficients flattened

    ker_dic: dict
             Kernels and their parameters
             keys : {1, ..., L, 'dim_tot'}
             values: dict with kernel info / sum of intermediate spaces dims

    Returns
    -------
    Phi_dic: dict
             Coefficients organized in layers
             keys: {1, ..., L-1, (L)} depending on implicitness of last layer
             values: arrays of shape (n_samples, n_features_l)
    """
    # Get constants
    L = len(ker_dic.keys()) - 1
    n = len(Phi_ravel) // ker_dic['dim_tot']

    # Create the separators that will split the ravel
    Phi_dic = {}
    sep = [0]

    # Loop over layers, select the corresponding coef in the ravel and reshape.
    # No .copy(), only a pointer. If dict value is modified, so is Phi_ravel.
    for l in range(1, L + 1):
        try:
            sep.append(sep[-1] + n * ker_dic[l]['outdim'])
            Phi_dic[l] = Phi_ravel[sep[l - 1]: sep[l]].view(
                n, ker_dic[l]['outdim']).requires_grad_(True)
        # Subtlety at last layer for OKAE (outdim = 'infty')
        except TypeError:
            pass

    return Phi_dic


###############################################################################

#                            Gram functions

###############################################################################


def poly_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    """Compute polynomial Gram matrix between X and Y (or X)

    Parameters
    ----------
    X: torch.Tensor of shape (n_samples_1, n_features)
       First input on which Gram matrix is computed

    Y: torch.Tensor of shape (n_samples_2, n_features), default None
       Second input on which Gram matrix is computed. X is reused if None

    degree: int
            Degree parameter of the kernel (see sklearn implementation)

    gamma: float
           Gamma parameter of the kernel

    coef0: float
           coef0 parameter of the kernel

    Returns
    -------
    K: torch.Tensor of shape (n_samples_1, n_samples_2)
       Gram matrix on X/Y
    """
    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K_tmp = torch.mm(X, torch.t(Y))
    K_tmp *= gamma
    K_tmp += coef0
    K = K_tmp ** degree

    return K


def rbf_kernel(X, Y=None, gamma=None):
    """Compute rbf Gram matrix between X and Y (or X)

    Parameters
    ----------
    X: torch.Tensor of shape (n_samples_1, n_features)
       First input on which Gram matrix is computed

    Y: torch.Tensor of shape (n_samples_2, n_features), default None
       Second input on which Gram matrix is computed. X is reused if None

    gamma: float
           Gamma parameter of the kernel (see sklearn implementation)

    Returns
    -------
    K: torch.Tensor of shape (n_samples_1, n_samples_2)
       Gram matrix on X/Y
    """
    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    X_norm = (X ** 2).sum(1).view(-1, 1)
    Y_norm = (Y ** 2).sum(1).view(1, -1)
    K_tmp = X_norm + Y_norm - 2. * torch.mm(X, torch.t(Y))
    K_tmp *= -gamma
    K = torch.exp(K_tmp)

    return K


def compute_Gram(k_dic, X, Y=None):
    """Compute the Gram matrix of an individual kernel

    Parameters
    ----------
    k_dic: dict
           Kernel and its parameter
           keys: {'kernel', 'kparam', 'lambda', 'outdim'}
           values: kernel type, parameter, regularization output dimension

    X: torch.Tensor of shape (n_samples_1, n_features)
       First input on which Gram matrix is computed

    Y: torch.Tensor of shape (n_samples_2, n_features), default None
       Second input on which Gram matrix is computed. X is reused if None

    Returns
    -------
    K: torch.Tensor of shape (n_samples_1, n_samples_2)
       Gram matrix on X/Y
    """
    # Compute Gram matrix depending on the kernel passed
    if k_dic['kernel'] == 'gaussian':
        K = rbf_kernel(X, Y=Y, gamma=k_dic['kparam'])

    elif k_dic['kernel'] == 'polynomial':
        K = poly_kernel(X, Y=Y, degree=k_dic['kparam'][0],
                        gamma=k_dic['kparam'][1], coef0=k_dic['kparam'][2])

    else:
        raise ValueError("Invalid kernel %s" % k_dic['kernel'])

    return K


def compute_Reprs_Grams(Phi_dic, ker_dic, X=None, G_in=None):
    """Compute the intermediate representations/Gram matrices from X or G_in

    Parameters
    ----------
    Phi_dic: dict
             Coefficients organized in layers
             keys: {1, ..., L-1, (L)} depending on dimension of last layer
             values: arrays of shape (n_samples, n_features_l)

    ker_dic: dict
             Kernels and their parameters
             keys : {1, ..., L, 'dim_tot'}
             values: dict with kernel info / sum of intermediate spaces dims

    X: torch.Tensor of shape (n_samples, n_features_0), default None
       Input representation. None if implicit representation via Gram matrix

    G_in: torch.Tensor of shape (n_samples, n_samples), default None
          Input Gram matrix of 1st layer. Necessary in implicit case, avoid re-
          computation in standard case

    Returns
    -------
    Reprs: dict
           Intermediate representations
           keys: {(0), 1, ..., L-1, (L)} depending on dim of 1st/last layer
           values: torch.Tensor of shape (n_samples, n_features_l)

    Grams: dict
           Intermediate Gram matrices
           keys: {1, ..., L}
           values: torch.Tensor of shape (n_samples, n_samples)
    """
    L = len(ker_dic.keys()) - 1

    try:
        Reprs = {0: X.clone()}
    except AttributeError:
        Reprs = {}

    if G_in is not None:
        Grams = {1: G_in.clone()}
    else:
        Grams = {1: compute_Gram(ker_dic[1], X)}

    # Recursive computation
    for l in range(1, L + 1):

        # No need to compute Gram at first layer (computed at initialization)
        if l > 1:
            Grams[l] = compute_Gram(ker_dic[l], Reprs[l - 1])

        try:
            try:
                Reprs[l] = torch.mm(Grams[l], torch.mm(Phi_dic[l],
                                                       ker_dic[l]['outmat']))
            # If no outmat, means torch.eye, simpler formula
            except KeyError:
                Reprs[l] = torch.mm(Grams[l], Phi_dic[l])

        # OKAE exception
        except KeyError:
            pass

    return Reprs, Grams


def compute_Reprs_Grams_te(Phi_dic, ker_dic, Reprs_tr, X_te=None,
                           G_in_te=None):
    """Compute intermediate Reprs and final Gram from X_te and/or G_in_te

    Parameters
    ----------
    Phi_dic: dict
             Coefficients organized in layers
             keys: {1, ..., L-1, (L)} depending on dimension of last layer
             values: arrays of shape (n_samples, n_features_l)

    ker_dic: dict
             Kernels and their parameters
             keys : {1, ..., L, 'dim_tot'}
             values: dict with kernel info / sum of intermediate spaces dims

    Reprs_tr: dict
              Intermediate representations of train data
              keys: {(0), 1, ..., L-1, (L)} depending on dim of 1st/last layer
              values: torch.Tensor of shape (n_samples, n_features_l)

    X_te: torch.Tensor of shape (n_samples, n_features_0), default None
          Test Input representation. None if implicit representation

    G_in_te: torch.Tensor of shape (n_samples, n_samples), default None
             Test input Gram matrix of first layer. Necessary in implicit case,
             avoid re-computation in standard one

    Returns
    -------
    Reprs_te: dict
              Intermediate test representations
              keys: {(0), 1, ..., L-1, (L)} depending on dim of 1st/last layer
              values: torch.Tensor of shape (n_samples, n_features_l)

    Grams_te: dict
              Intermediate test Gram matrices
              keys: {1, ..., L}
              values: torch.Tensor of shape (n_samples, n_samples)
    """
    L = len(ker_dic.keys()) - 1

    try:
        Reprs_te = {0: X_te.clone()}
    except AttributeError:
        Reprs_te = {}

    if G_in_te is not None:
        Gram_te = G_in_te.clone()
    else:
        Gram_te = compute_Gram(ker_dic[1], X_te, Y=Reprs_tr[0])

    # Recursive computation
    for l in range(1, L + 1):

        # No need to compute Gram at first layer (computed at initialization)
        if l > 1:
            Gram_te = compute_Gram(ker_dic[l], Reprs_te[l - 1],
                                   Y=Reprs_tr[l - 1])

        try:
            try:
                Reprs_te[l] = torch.mm(Gram_te, torch.mm(Phi_dic[l],
                                                         ker_dic[l]['outmat']))
            # If no outmat, means torch.eye, simpler formula
            except KeyError:
                Reprs_te[l] = torch.mm(Gram_te, Phi_dic[l])

        # OKAE exception
        except KeyError:
            pass

    return Reprs_te, Gram_te


###############################################################################

#                          Objective functions

###############################################################################


def layer_pen(Phi, k_dic, Gram):
    """Compute layer's penalization from layer's information

    Parameters
    ----------
    Phi: torch.Tensor of shape (n_samples, n_features_l+1)
         Coefficient tensor

    k_dic: dict
           Kernel and its parameter
           keys: {'kernel', 'kparam', 'lambda', 'outdim'}
           values: kernel type, parameter, regularization output dimension

    Gram: torch.Tensor of shape (n_samples, n_samples)
          Gram matrix

    Returns
    -------
    pen: float
         Penalization value
    """
    # Simpler formula if outmat = torch.eye
    try:
        N = torch.mm(Phi, torch.mm(k_dic['outmat'], torch.t(Phi)))
    except KeyError:
        N = torch.mm(Phi, torch.t(Phi))

    pen = k_dic['lambda'] * (Gram * N).sum()

    return pen


def o_layer_pen(N_L, Gram_L, lambda_L):
    """Compute last layer penalization

    Parameters
    ----------
    N_L: torch.Tensor of shape (n_samples, n_samples)
         Last layer coefficients' dot products

    Gram_L: torch.Tensor of shape (n_samples, n_samples)
            Last layer Gram matrix

    lambda_L: float
              Last layer regularization parameter

    Returns
    -------
    pen: float
         Penalization value
    """
    pen = lambda_L * (Gram_L * N_L).sum()
    return pen


def MSD(Z, Y):
    """Compute the mean square distortion (MSD) between Z and Y

    Parameters
    ----------
    Z: torch.Tensor of shape (n_samples, n_features)
       Tensor to be compared

    Y: torch.Tensor of shape (n_samples, n_features)
       Tensor to be compared

    Returns
    -------
    msd: float
         Mean square distance between Z and Y
    """
    msd = ((Z - Y) ** 2).sum()
    msd /= Z.shape[0]
    return msd


def o_MSD(N_L, lambda_L):
    """Compute MSD (after KRR)

    Parameters
    ----------
    N_L: torch.Tensor of shape (n_samples, n_samples)
         Last layer coefficients' dot products

    lambda_L: float
              Last layer regularization parameter

    Returns
    -------
    msd: float
         (Implicit) mean square distance between last layer output and Y
    """
    msd = N_L.shape[0] * lambda_L ** 2 * N_L.trace()
    return msd


def o_MSD_te(Phi_dic, N_L, ker_dic, Reprs_tr, G_tr_L, G_in_te, G_out_te,
             G_out_own_te):
    """Compute test distortion

    Parameters
    ----------
    Phi_dic: dict
             Coefficients organized in layers
             keys: {1, ..., L-1, (L)} depending on dimension of last layer
             values: arrays of shape (n_samples, n_features_l)

    N_L: torch.Tensor of shape (n_samples, n_samples)
         Last layer coefficients' dot products

    ker_dic: dict
             Kernels and their parameters
             keys : {1, ..., L, 'dim_tot'}
             values: dict with kernel info / sum of intermediate spaces dims

    Reprs_tr: dict
              Intermediate representations of train data
              keys: {(0), 1, ..., L-1, (L)} depending on dim of 1st/last layer
              values: torch.Tensor of shape (n_samples_tr, n_features_l)

    G_tr_L: torch.Tensor of shape (n_samples_tr, n_samples_tr)
            Last layer train Gram matrix

    G_in_te: torch.Tensor of shape (n_samples_tr, n_samples_te)
             First layer test Gram matrix

    G_out_te: torch.Tensor of shape (n_samples_tr, n_samples_te)
              Test output Gram matrix

    G_out_own_te: torch.Tensor of shape (n_samples_te, n_samples_te)
                  Test-test output Gram matrix

    Returns
    -------
    msd: float
         Test (implicit) mean square distance
    """
    L = len(ker_dic.keys()) - 1

    Reprs_te, Gram_te = compute_Reprs_Grams_te(Phi_dic, ker_dic, Reprs_tr,
                                               G_in_te=G_in_te)

    n_tr = G_tr_L.shape[0]
    W = G_tr_L + n_tr * ker_dic[L]['lambda'] * torch.eye(n_tr,
                                                         dtype=torch.float64)
    W_inv = torch.inverse(W)

    A = torch.trace(G_out_own_te)
    B = (torch.mm(Gram_te, N_L) * Gram_te).sum()
    C = -2 * (torch.mm(Gram_te, W_inv) * G_out_te).sum()

    # Return the distortion
    msd = 1. / Gram_te.shape[0] * (A + B + C)
    return msd


def objective(Phi_dic, ker_dic, N_L=None, X=None, G_in=None, Y=None,
              Reprs=None, Grams=None):
    """Compute objective

    Parameters
    ----------
    Phi_dic: dict
             Coefficients organized in layers
             keys: {1, ..., L-1, (L)} depending on dimension of last layer
             values: arrays of shape (n_samples, n_features_l)

    ker_dic: dict
             Kernels and their parameters
             keys : {1, ..., L, 'dim_tot'}
             values: dict with kernel info / sum of intermediate spaces dims

    N_L: torch.Tensor of shape (n_samples, n_samples), default None
         Last layer coefficients' dot products. Useless in standard case

    X: torch.Tensor of shape (n_samples, n_features_0), default None
       Input representation. None if implicit representation via Gram matrix

    G_in: torch.Tensor of shape (n_samples, n_samples), default None
          Input Gram matrix of 1st layer. Necessary in implicit case, avoid re-
          computation in standard case

    Y: torch.Tensor of shape (n_samples, n_features_L), default None
       Target. None in implicit case, information already contained in N_L

    Reprs: dict
           Intermediate representations
           keys: {(0), 1, ..., L-1, (L)} depending on dim of 1st/last layer
           values: torch.Tensor of shape (n_samples, n_features_l)

    Grams: dict
           Intermediate Gram matrices
           keys: {1, ..., L}
           values: torch.Tensor of shape (n_samples, n_samples)

    Returns
    -------
    obj: float
         Objective = (implicit) msd + penalizations
    """
    L = len(ker_dic.keys()) - 1
    obj = 0

    # If Reprs and Grams not pre-computed, compute them
    if Reprs is None and Grams is None:
        Reprs, Grams = compute_Reprs_Grams(Phi_dic, ker_dic, X=X, G_in=G_in)

    # Add internal layers penalizations
    for l in range(1, L):
        obj += layer_pen(Phi_dic[l], ker_dic[l], Grams[l])

    # Difference on MSD / last layer penalization
    if ker_dic[L]['outdim'] == 'infty':
        obj += o_layer_pen(N_L, Grams[L], ker_dic[L]['lambda'])
        obj += o_MSD(N_L, ker_dic[L]['lambda'])

    else:
        obj += layer_pen(Phi_dic[L], ker_dic[L], Grams[L])
        obj += MSD(Reprs[L], Y)

    return obj


###############################################################################

#                          Alternate functions

###############################################################################


def kron(t1, t2):
    """
    Compute Kronecker product in pytorch between two tensors (not optimized)

    Parameters
    ----------
    t1: torch.Tensor of size (h1, w1)
        First matrix of Kronecker product

    t2: torch.Tensor of size (h2, w2)
        Second matrix of Kronecker product

    Returns
    -------
    t3: torch.Tensor of size (h1 * h2, w1 * w2)
        Kronecker product of t1 and t2
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height, out_width = t1_height * t2_height, t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (t1.unsqueeze(2).unsqueeze(3).
                   repeat(1, t2_height, t2_width, 1).
                   view(out_height, out_width))

    t3 = expanded_t1 * tiled_t2
    return t3


def compute_Phi_L(k_dic_L, Gram_L, Y):
    """Compute Phi_L using KRR

    Parameters
    ----------
    k_dic_L: dict
             Last layer kernel and its parameter
             keys: {'kernel', 'kparam', 'lambda', 'outdim'}
             values: kernel type, parameter, regularization output dimension

    Gram_L: torch.Tensor of shape (n_samples, n_samples)
            Last layer Gram matrix

    Y: torch.Tensor of shape (n_samples, n_features_L)
       Target

    Returns
    -------
    Phi_L: torch.Tensor of size (n_samples, n_features_L)
           Optimal last coefficient computed thanks to KRR
    """
    n, d = Y.shape

    # Simpler formula if output matrix is torch.eye
    try:
        Gram_Tot = kron(Gram_L, k_dic_L['outmat'])
        M = Gram_Tot + n * k_dic_L['lambda'] * torch.eye(n * d,
                                                         dtype=torch.float64)
        Phi_L, _ = torch.gesv(Y.flatten(), M)
        Phi_L = Phi_L.view(n, -1)
    except KeyError:
        M = Gram_L + n * k_dic_L['lambda'] * torch.eye(n, dtype=torch.float64)
        Phi_L, _ = torch.gesv(Y, M)

    # Should be independent from previous layers: .data
    return Phi_L.data


def compute_N_L(lambda_L, Gram_L, G_out):
    """Compute N_L using KRR

    Parameters
    ----------
    lambda_L: float
              Last layer regularization parameter

    Gram_L: torch.Tensor of shape (n_samples, n_samples)
            Last layer Gram matrix

    G_out: torch.Tensor of shape (n_samples, n_samples)
           Output Gram matrix

    Returns
    -------
    N_L: torch.Tensor of shape (n_samples, n_samples), default None
         Optimal last layer coefficients' dot products computed thanks to KRR
    """
    n = Gram_L.shape[0]
    M_L = Gram_L + n * lambda_L * torch.eye(n, dtype=torch.float64)
    M_L_1 = torch.inverse(M_L)
    N_L = torch.mm(M_L_1, torch.mm(G_out, M_L_1))

    # Should be independent from previous layers: .data
    return N_L.data


###############################################################################

#                                 Class

###############################################################################


class KAE:
    """K(2)AE class with fitting (optimization) procedures in pytorch
    """

    def __init__(self, kernels, kparams, outmats, lambdas):
        self.ker_dic = mk_ker_dic(kernels, kparams, outmats, lambdas)
        self.L = len(self.ker_dic.keys()) - 1

    def set_Phi(self, Phi_ravel):
        """Set Phi_dic to specified value
        """
        self.Phi_dic = mk_Phi_dic(Phi_ravel.clone(), self.ker_dic)

    def compute_RG_tr(self, X=None, G_in=None):
        """Compute intermediate Reprs/Grams from X and/or G_in
        """
        self.Reprs_tr, self.Grams_tr = compute_Reprs_Grams(
            self.Phi_dic, self.ker_dic, X=X, G_in=G_in)

    def set_PhiL_KRR(self, Y):
        """Compute optimal KRR Phi_L
        """
        self.Phi_dic[self.L] = compute_Phi_L(self.ker_dic[self.L],
                                             self.Grams_tr[self.L], Y)

    def set_NL(self, NL='auto', G_in=None, G_out=None):
        """Compute KRR optimal N_L, or set arbitrary one
        """
        if isinstance(NL, torch.Tensor):
            self.N_L = NL.clone()

        elif NL == 'auto':
            # If no Reprs_tr, need to compute them (and a G_in to do it)
            if not hasattr(self, 'Reprs_tr'):
                if G_in is None:
                    raise ValueError('No train representations'
                                     ', need fitting or G_in')
                else:
                    self.compute_RG_tr(G_in=G_in)

            # Finally, compute N_L using KRR
            self.N_L = compute_N_L(self.ker_dic[self.L]['lambda'],
                                   self.Grams_tr[self.L], G_out)

    def fit(self, X=None, Y=None, G_in=None, G_out=None, method='gd',
            n_epoch=100, n_loop=10, solver='sgd', **kwargs):
        """Fit best parameters Phi_dic from X and Y or G_in and G_out

        Parameters
        ----------
        X: torch.Tensor of shape (n_samples, n_features_0), default None
           Input. None in the implicit case as given via an input Gram matrix

        Y: torch.Tensor of shape (n_samples, n_features_L), default None
           Target. None in the implicit case as given via an output Gram matrix

        G_in: torch.Tensor of shape (n_samples, n_samples), default None
              Input Gram matrix. Necessary in implicit case. Avoid re-
              computation in standard one

        G_out: torch.Tensor of shape (n_samples, n_samples), default None
               Output Gram matrix. Useless in standard case

        method: str, default 'gd'
                Method to use for the optimization

        n_epoch: int, default 100
                 Number of epochs in inner solver

        n_loop: int, default 10
                Number of epochs for inner + outer update

        solver: str, default 'sgd'
                Inner solver to use

        **params: kwargs
                  keyword arguments passed to inner solver
        """
        # Test if optimization method is valid
        if method in ['gd', 'md']:
            if not isinstance(self.ker_dic[self.L]['outdim'], int):
                raise ValueError('Invalid optimization method called')

        elif method in ['ogd', 'omd']:
            if self.ker_dic[self.L]['outdim'] != 'infty':
                raise ValueError('Invalid optimization method called')

        else:
            raise ValueError('Invalid optimization method called')

        # Random initialization if no parameter
        if not hasattr(self, 'Phi_dic'):
            try:
                n = X.shape[0]
            except TypeError:
                n = G_in.shape[0]
            Phi_init = torch.randn(n * self.ker_dic['dim_tot'],
                                   dtype=torch.float64)
            self.set_Phi(Phi_init)

        if method in ['gd', 'md']:
            self.N_L = None

        elif method in ['ogd', 'omd']:
            self.set_NL(G_in=G_in, G_out=G_out)

        # To store objective/time iterations
        if not hasattr(self, 'losses'):
            self.losses = []
            self.times = [0]

        # Coefficients to be updated (only L - 1 internal ones in md)
        if method == 'gd':
            params = (self.Phi_dic[l] for l in range(1, self.L + 1))

        elif method in ['md', 'ogd', 'omd']:
            params = (self.Phi_dic[l] for l in range(1, self.L))

        # Criterion to be optimized
        def closure():
            loss = objective(self.Phi_dic, self.ker_dic, N_L=self.N_L, X=X,
                             G_in=G_in, Y=Y)
            optimizer.zero_grad()
            loss.backward()
            return loss

        # Optimizer
        if solver == 'sgd':
            optimizer = optim.SGD(params, **kwargs)
        elif solver == 'lbfgs':
            optimizer = optim.LBFGS(params, **kwargs)
        else:
            raise ValueError('Invalid solver: %s' % solver)

        # By default (gd, ogd), only 1 outer loop
        n_outer_epoch = 1
        if method in ['md', 'omd']:
            n_outer_epoch = n_loop

        t0 = time.time() - self.times[-1]

        for k in range(n_outer_epoch):
            for t in range(n_epoch):

                # Coefficient update
                loss = closure()
                self.losses.append(loss.item())
                self.times.append(time.time() - t0)
                optimizer.step(closure)

            # Compute intermediate train Reprs/Grams
            self.compute_RG_tr(X=X, G_in=G_in)

            # Last layer update in mirror descents
            if method == 'md':
                self.set_PhiL_KRR(Y)
            elif method == 'omd':
                self.set_NL(G_in=G_in, G_out=G_out)

        # Clean times if necessary (i.e. remove first 0)
        try:
            self.times.remove(0)
        except ValueError:
            pass

    def predict(self, X_te=None, G_in_te=None):
        """Give prediction from X_te or G_in_te

        X_te: torch.Tensor of shape (n_samples, n_features_0), default None
              Test Input representation. None if implicit representation

        G_in_te: torch.Tensor of shape (n_samples, n_samples), default None
                 Test input Gram matrix of first layer. Necessary in implicit
                 case, avoid re-computation in standard one

        Returns
        -------
        Reprs_te: dict
                  Intermediate test representations
                  keys: {(0), 1, ..., L-1, (L)} depending on implicitness
                  values: torch.Tensor of shape (n_samples, n_features_l)
        """
        Reprs_te, _ = compute_Reprs_Grams_te(
            self.Phi_dic, self.ker_dic, self.Reprs_tr, X_te=X_te,
            G_in_te=G_in_te)

        return Reprs_te

    def test_disto(self, X_te=None, Y_te=None, G_in_te=None, G_out_te=None,
                   G_out_own_te=None):
        """Compute test distortion

        Parameters
        ----------
        X_te: torch.Tensor of shape (n_samples_te, n_features_0),
              default None
              Test input in non implicit case

        Y_te: torch.Tensor of shape (n_samples_te, n_features_L),
              default None
              Test target in non implicit case

        G_in_te: torch.Tensor of shape (n_samples_tr, n_samples_te),
                 default None
                 First layer test Gram matrix. For the implicit case

        G_out_te: torch.Tensor of shape (n_samples_tr, n_samples_te),
                  default None
                  Last layer test output Gram matrix. For the implicit case

        G_out_own_te: torch.Tensor of shape (n_samples_te, n_samples_te),
                      default None
                      Test-test output Gram matrix in the implicit case

        Returns
        -------
        disto: float
               Test distortion
        """
        Reprs_te, Gram_te = compute_Reprs_Grams_te(
            self.Phi_dic, self.ker_dic, self.Reprs_tr,
            X_te=X_te, G_in_te=G_in_te)

        try:
            disto = MSD(Y_te, Reprs_te[self.L])

        except KeyError:
            disto = o_MSD_te(
                self.Phi_dic, self.N_L, self.ker_dic, self.Reprs_tr,
                self.Grams_tr[self.L], G_in_te, G_out_te, G_out_own_te)

        return disto

    def clear_memory(self):
        """Clear fitting memory
        """
        self.losses, self.times = [], [0]
