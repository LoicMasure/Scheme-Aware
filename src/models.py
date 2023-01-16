"""
Module for building neural network models.
Author: Loïc Masure
Date: 21/04/2021
"""
from typing import Tuple, Dict, List
import torch
from torchvision import transforms
import hadamard_cuda
import numpy as np
import copy

import utils


# ################ New functions that serve as param-free layers ##############
def convolve_2(x: torch.Tensor,
               y: torch.Tensor,
               scheme: str = "boolean"
               ) -> torch.Tensor:
    """
    Vanilla convolution function for Boolean masking.
    Only works for tensors of shape (-, length) where length is a power of 2.
    """
    z = torch.zeros(*x.shape)
    length = x.shape[1]
    assert(utils.is_power_of_two(length))
    op = utils.group_law(scheme, length)

    for i in range(z.shape[1]):
        for j in range(z.shape[1]):
            z[:, i] += x[:, j] * y[:, op(i, j)]
    return z


def convolve(x: torch.Tensor, scheme: str = "boolean") -> torch.Tensor:
    """
    Expands the former naive convolution for more than two shares.
    Assumes input of shape (batch_size, n_shares, length)
    """
    prod = x[:, 0, :].clone()
    for i in range(1, x.shape[1]):
        prod = convolve_2(prod, x[:, i, :], scheme)
    return prod


class FFWHT(torch.autograd.Function):
    """
    Source: pytorch forum (put the link)
    """
    @staticmethod
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the Walsh-Hadamard transform on the last dimension.
        The normalization depends on the number of shares.
        Input:
            * tensor: tensor of shape (..., #shares, length)
        Output:
            * tensor: tensor of shape (..., #shares, length)
        """
        length = tensor.shape[-1]
        n_shares = tensor.shape[-2]
        n_bits = int(np.log2(length))
        result = hadamard_cuda.hadamard_transform(tensor)
        result /= 2 ** (n_bits / n_shares / 2)
        return result


    @staticmethod
    def forward(ctx, input):
        return FFWHT.transform(input)

    @staticmethod
    def backward(ctx, grad_output):
        return FFWHT.transform(grad_output)


def convolve_WH(x: torch.Tensor) -> torch.Tensor:
    """
    Convolution with the Walsh-Hadamard transform.
    """
    transform = FFWHT.apply  # transform = FWHT.transform
    length = x.shape[-1]
    assert(utils.is_power_of_two(length))

    x_wh = transform(x)
    prod = x_wh.prod(dim=1, keepdim=True)
    res = transform(prod)[:, 0, :]

    return res


def prod_(x: torch.Tensor, dim=1) -> torch.Tensor:
    """
    Applies the element-wise product along a dimension.
    Necessary until the complex autograd function is implemented
    in Pytorch.
    Sources:
    discuss.pytorch.org/t/fourier-transform-and-complex-dtype-restrictions/
    https://github.com/pytorch/pytorch/pull/48125
    """
    res = x.select(dim, 0)
    for i in range(1, x.size(dim)):
        res = res * x.select(dim, i)
    return res


def convolve_fft(x: torch.Tensor) -> torch.Tensor:
    """
    Convolution with the Fast Fourier Transform (FFT).
    """
    x_fft = torch.fft.fft(x, dim=-1)
    prod_fft = prod_(x_fft, dim=1)
    return torch.fft.ifft(prod_fft, dim=-1).real


def convolve_mult(x: torch.Tensor) -> torch.Tensor:
    """
    x: Tensor of shape (batch_size, n_shares, length)
    The (n_shares - 1) 1st shares are multiplicative masks (strictly positive)
    The last share denotes the masked data.
    """
    n_bits = int(np.log2(x.shape[-1]))
    log, alog = utils.get_log_tables(n_bits)
    # Turns to log space
    x_alog = x[:, :, alog[1:]]
    # Convolution like with arithmetical scheme
    prod_alog = convolve_fft(x_alog)
    # Turns back from log space
    log_np = np.array(log)
    prod = prod_alog[:, log_np-2]  # Don't know why -2 ...
    # Adds the case where the masked data is 0, bc probs not normalized yet
    res_0 = x[:, -1, :1] * x[:, :-1, :].sum(dim=-1, keepdim=True).prod(dim=1)
    return torch.cat((res_0, prod[:, 1:]), dim=-1)


def convolve_affine_naive(alpha, masked, beta):
    res = torch.zeros(masked.shape, device=device)
    for s in range(masked.shape[-1]):
        for a in range(1, masked.shape[-1]):
            for b in range(masked.shape[-1]):
                idx = multGF(s, a, log, alog) ^ b
                res[:, s] += masked[:, idx] * alpha[:, a] * beta[:, b]
    return res

def convolve_affine(
        alphas: torch.Tensor,
        masked: torch.Tensor,
        betas: torch.Tensor
        ) -> torch.Tensor:
    """
    Performs the convolution for the affine masking.
    NB : as it is implemented in the ASCAD paper, one should use cross-correlation
    instead of convolutions !
    Dimensions:
        * alphas, betas: [batch_size, (n_shares-1)//2, 2^n]
        * masked: [batch_size, 1, 2^n]
    """
    betas = torch.cat((masked[:, None, :], betas), dim=1)
    betas = convolve_WH(betas)

    # Turn to log space
    n_bits = int(np.log2(masked.shape[-1]))
    log, alog = utils.get_log_tables(n_bits)
    betas_alog = betas[:, alog[:-1]]
    alphas_alog = alphas[:, :, alog[:-1]]

    # Applies FFT
    betas_fft = torch.fft.fft(betas_alog, dim=-1)
    alphas_fft = torch.fft.fft(alphas_alog, dim=-1)
    alphas_fft = torch.conj(alphas_fft)  # Important to take the conjugate here
    alphas_prod = prod_(alphas_fft, dim=1)
    fft_prod = alphas_prod * betas_fft
    prod_alog = torch.fft.ifft(fft_prod, dim=-1).real

    # Turn back from log space
    log_np = np.array(log[1:])
    prod = prod_alog[:, log_np]  # Don't know why -2 ...

    # Adds the case where the masked data is 0
    res_0 = betas[:, :1] * alphas[:, :, 1:].sum(dim=-1, keepdim=True).prod(dim=1)
    return torch.cat((res_0, prod), dim=-1)



###############################################################################


# ######################### Modules of Neural Nets ############################
class Branch(torch.nn.Module):
    """
    A class defining the NNs that fit each elementary leakage.
    """
    def __init__(self, dim_input: int, n_hidden: int, n_bits: int):
        """
        We declare here all the layers that carry learning parameters (no
        activation layer).
        Params:
            * n_hidden (int): number of neurons in the hidden layer
            * n_bits: (int): determines the number of classes to 

        NB: don't remove bias in the linear layer !! (don't know why...)
        """
        super(Branch, self).__init__()

        # Between input and hidden layer
        self.bn_0 = torch.nn.BatchNorm1d(dim_input)
        self.fc_1 = torch.nn.Linear(dim_input, n_hidden)

        # Between hidden layer and probability vector
        self.bn_1 = torch.nn.BatchNorm1d(n_hidden)
        self.fc_2 = torch.nn.Linear(n_hidden, 1 << n_bits)

    def forward(self, leakage: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the branch.
        Params:
            * leakage: tensor of shape (batch_size, dim_input)
        Returns:
            * scores: tensor of shape (batch_size, 2^n_bits)
        """
        # First layer
        x = self.bn_0(leakage)
        x = self.fc_1(x)
        x = torch.nn.functional.relu(x)

        # Second layer
        x = self.bn_1(x)
        x = self.fc_2(x)

        return x

###################################################################
class Recombine(torch.nn.Module):
    """
[train_data.leakages[k].shape for k in train_data.leakages.keys()]
    A class to instanciate recombination modules.
    """
    def __init__(self,
                 keys_leakages: List[str],
                 n_bits: int = 8,
                 normalize_probs: bool = True):
        super(Recombine, self).__init__()
        self.normalize_probs = normalize_probs
        self.n_bits = n_bits
        self.dict_leakages = keys_leakages
    
    def __repr__(self):
        return "Recombine: {}".format(self.dict_leakages)

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.softmax(x, dim=-1)  # Maybe try another norm ?
        assert(utils.isProbDist(x))
        return x
    
    def forward(self, leakages: Dict[str, torch.Tensor]
            ) -> Dict[str, torch.Tensor]:
        """
        The forward pass (the backward is automatically derived).
        """
        raise NotImplementedError


class GroupRecombine(Recombine):
    """
    Recombination with convolutions w.r.t. a unique inner law, e.g. Boolean,
    arithmetical, multiplicative, but not affine.
    """
    def __init__(self, keys_leakages: List[str],
                 n_bits: int = 8,
                 scheme: str = "boolean",
                 normalize_probs: bool = False):
        super(GroupRecombine, self).__init__(keys_leakages, n_bits,normalize_probs)
        self.scheme = scheme
        if self.scheme == "boolean":
            self.conv = convolve_WH
        elif self.scheme == "arithmetical":
            self.conv = convolve_fft
        elif self.scheme == "multiplicative":
            self.conv = convolve_mult
        else:
            self.conv = lambda x: convolve(x, scheme=self.scheme)
        self.tol = 1e-5
    
    def __repr__(self):
        return super().__repr__() + " with {}".format(self.conv)


    def forward(self, leakages: Dict[str, torch.Tensor]) -> Dict[str,
        torch.Tensor]:
        # Computes the log_probas for all the shares
        log_probas = {
                k: torch.nn.functional.log_softmax(val, dim=-1) 
                for k, val in leakages.items()
                }
        
        # Softmax normalization before convolution if specified
        if self.normalize_probs:
            leakages = {k: self.normalize(val) for k, val in leakages.items()}

        # Splits the distros
        p_beta = torch.stack([
            val
            for k, val in leakages.items() if "beta" in k
            ], dim=1)

        p_masked = {
                k: val
                for k, val in leakages.items() if "masked" in k
                }

        # Applies the discrete convolution to recombine from the shares
        for masked_key, target in p_masked.items():
            cat = torch.cat((p_beta, target[:, None, :]), dim=1)
            res = self.conv(cat)
            if self.normalize_probs:
                res = torch.clip(res, min=self.tol, max=1)
                res = torch.log(res)
            else:
                res = torch.nn.functional.log_softmax(res, dim=-1)
            # Sets the right name for the target key
            target_key = "target_" + masked_key.split("_")[1]
            # Completes the log_probas dict with targets
            log_probas[target_key] = res

        return log_probas


class GroupNet(torch.nn.Module):
    """
    """    
    def __init__(self,
                 leakage_dims: Dict[str, int],
                 n_hidden: int = 1000,
                 scheme: str = "boolean",
                 n_bits: int = 8,
                 normalize_probs: bool = False,
                 share_weights: bool = False,
                 known_PoIs: bool = True,
                 **kwargs):
        """
        Params:
            * shares_dims: dict of dimensionality for each leakage.
            * scheme: type of masking scheme (boolean, arithmetical, etc)
        """
        super(GroupNet, self).__init__()
        self.scheme = scheme
        self.n_bits = n_bits
        self.n_hidden = n_hidden
        self.normalize_probs = normalize_probs
        self.known_PoIs = known_PoIs

        # Sets the leakage dimensionality depending on the known PoIs or not
        if not self.known_PoIs:
            self.leakage_dims = {
                    k: sum(leakage_dims.values()) 
                    for k, val in leakage_dims.items()
                    }
        else:
            self.leakage_dims = leakage_dims

        # Instantiates the branches
        if share_weights:
            max_dim = np.max(list(self.leakage_dims.values()))
            only_branch_ = Branch(
                    max_dim, 
                    self.n_hidden, 
                    self.n_bits, 
                    )
            branches_ = {k: only_branch_ for k, val in self.leakage_dims.items()}
        else:
            branches_ = {
                    k: Branch(
                        val, 
                        self.n_hidden, 
                        self.n_bits, 
                        ) 
                    for k, val in self.leakage_dims.items()
                    }
        self.branches = torch.nn.ModuleDict(branches_)
        
        # Instantiates the masking encoding
        if self.scheme != "affine":
            self.recombine = GroupRecombine(
                    list(self.leakage_dims.keys()), 
                    n_bits=self.n_bits, 
                    normalize_probs=self.normalize_probs, 
                    scheme=self.scheme
                    )
        elif self.scheme == "affine":
            self.recombine = ASCADRecombine(
                    list(self.leakage_dims.keys()),
                    n_bits=self.n_bits,
                    normalize_probs=self.normalize_probs,
                    )

    def forward(self, leak_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.known_PoIs:
            full_leak = torch.hstack([val for k, val in leak_dict.items()])
            leak_dict = {k: full_leak for k, val in leak_dict.items()}

        # Forward pass on each 
        branch_res = {k: self.branches[k](val) for k, val in leak_dict.items()}
        # if self.normalize_probs:
        #     for k,val in branch_res.items():
        #         assert(utils.isProbDist(val))
        # Forward pass for recombination
        return self.recombine(branch_res)



############################ FOR BENCHMARK ####################################
class BBMLP(torch.nn.Module):
    """
    A class of Models that implements 
    """
    def __init__(
            self,
            leakage_dims: Dict[str, int],
            n_hidden: int = 1000,
            n_bits: int = 8,
            ):
        super(BBMLP, self).__init__()
        self.leakage_dim = sum(leakage_dims.values())
        self.n_hidden = n_hidden
        self.n_bits = n_bits

        # Splits the distros
        dim_masked = {k: val for k, val in leakage_dims.items() if "masked" in k}

        self.branches_ = {
                k: Branch(
                    self.leakage_dim, 
                    n_hidden=self.n_hidden, 
                    n_bits=self.n_bits, 
                    )
                for k, val in dim_masked.items()
                }
        self.branches = torch.nn.ModuleDict(self.branches_)

    def forward(self, leak_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the BB model.
        """
        stacked_leakage = torch.hstack([val for k, val in leak_dict.items()])
        log_probas = dict()
        for k, val in self.branches.items():
            tmp = val(stacked_leakage)
            target_key = "target_" + k.split("_")[1]
            log_probas[target_key] = torch.nn.functional.log_softmax(tmp, dim=-1)
        return log_probas



class NaiveRecombine(Recombine):
    """
    Recombination with convolutions w.r.t. a unique inner law, e.g. Boolean,
    arithmetical, multiplicative, but not affine.
    """
    def __init__(self, keys_leakages: List[str],
                 n_bits: int = 8,
                 n_shares: int = 2,
                 n_hidden: int = 100,
                 normalize_probs: bool = False):
        super(NaiveRecombine, self).__init__(keys_leakages, n_bits,normalize_probs)
        self.n_bits = n_bits
        self.n_shares = n_shares
        self.n_hidden = n_hidden
        self.fc_1 = torch.nn.Linear(self.n_shares * (1 << self.n_bits), self.n_hidden)
        self.fc_2 = torch.nn.Linear(self.n_hidden, 1 << self.n_bits)
    
    def __repr__(self):
        return super().__repr__() + " with {}, {}".format(self.fc_1, self.fc_2)

    def forward(self, leakages: Dict[str, torch.Tensor]) -> Dict[str,
        torch.Tensor]:
        # Computes the log_probas for all the shares
        log_probas = {
                k: torch.nn.functional.log_softmax(val, dim=-1) 
                for k, val in leakages.items()
                }

        # Softmax normalization before recombination if specified
        if self.normalize_probs:
            leakages = {k: self.normalize(val) for k, val in leakages.items()}
        
        # Splits the distros
        p_beta = torch.stack([
            val
            for k, val in leakages.items() if "beta" in k
            ], dim=1)
        p_masked = {
                k: val
                for k, val in leakages.items() if "masked" in k
                }

        # Applies the discrete convolution to recombine from the shares
        for masked_key, target in p_masked.items():
            cat = torch.cat((p_beta, target[:, None, :]), dim=1)
            ################ MLP ###############
            cat = cat.reshape(-1, self.n_shares * (1 << self.n_bits))
            res = self.fc_1(cat)
            res = torch.nn.functional.relu(res)
            res = self.fc_2(res)
            ####################################
            res = torch.nn.functional.log_softmax(res, dim=-1)
            # Sets the right name for the target key
            target_key = "target_" + masked_key.split("_")[1]
            log_probas[target_key] = res

        return log_probas

class KnownPoI(torch.nn.Module):
    """
    Model that separates the treatment of the leakages, but modelizes the 
    recombination with MLP.
    """    
    def __init__(self,
                 leakage_dims: Dict[str, int],
                 n_hidden: int = 1000,
                 n_shares: int = 2,
                 n_bits: int = 8,
                 normalize_probs: bool = False,
                 share_weights: bool = False,
                 **kwargs):
        """
        Params:
            * shares_dims: dict of dimensionality for each leakage.
            * scheme: type of masking scheme (boolean, arithmetical, etc)
        """
        super(KnownPoI, self).__init__()
        self.n_shares = n_shares
        self.n_bits = n_bits
        self.leakage_dims = leakage_dims
        self.n_hidden = n_hidden
        self.normalize_probs = normalize_probs

        # Instantiates the branches
        if share_weights:
            max_dim = np.max(list(leakage_dims.values()))
            only_branch_ = Branch(max_dim, self.n_hidden, self.n_bits)
            branches_ = {k: only_branch_ for k, val in self.leakage_dims.items()}
        else:
            # branches_ = {k: LDA(val, self.n_bits) for k, val in self.leakage_dims.items()}
            branches_ = {
                    k: Branch(val, self.n_hidden, self.n_bits) 
                    for k, val in self.leakage_dims.items()
                    }
        self.branches = torch.nn.ModuleDict(branches_)
        
        # Instantiates the masking encoding
        self.recombine = NaiveRecombine(list(self.leakage_dims.keys()), n_bits=self.n_bits, n_shares=self.n_shares, n_hidden=100, normalize_probs=self.normalize_probs)

    def forward(self, leak_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Forward pass on each 
        branch_res = {k: self.branches[k](val) for k, val in leak_dict.items()}
        # Forward pass for recombination
        return self.recombine(branch_res)

########################## ASCAD ##############################################
class ASCADBranch(torch.nn.Module):
    """
    A subclass of Branch dedicated to ASCAD.
    """
    def __init__(
            self,
            dim_input: int, 
            n_filters_1: int,
            kernel_size: int,
            pool_size: int,
            n_bits: int
            ):
        super(ASCADBranch, self).__init__()
        # Sets the dimensionalities of the data (..., channels, dim)
        self.channels = 1
        self.dim = dim_input

        # Sets the first convolutional layer
        self.conv1 = torch.nn.Conv1d(
                in_channels=self.channels,
                out_channels=n_filters_1, 
                kernel_size=kernel_size, 
                padding=(kernel_size-1)//2,
                bias=False
                )
        self.channels = n_filters_1
        self.bn1 = torch.nn.BatchNorm1d(self.channels)
        self.pool1 = torch.nn.AvgPool1d(pool_size)
        self.dim //= pool_size

        # Sets the second convolutional layer
        self.conv2 = torch.nn.Conv1d(
                in_channels=self.channels,
                out_channels=self.channels * 2,
                kernel_size=kernel_size,
                padding=(kernel_size-1)//2,
                bias=False
                )
        self.channels *= 2
        self.bn2 = torch.nn.BatchNorm1d(self.channels)
        self.pool2 = torch.nn.AvgPool1d(pool_size)
        self.dim //= pool_size

        # Sets the third convolutional layer
        self.conv3 = torch.nn.Conv1d(
                in_channels=self.channels,
                out_channels=self.channels * 2,
                kernel_size=kernel_size,
                padding=(kernel_size-1)//2,
                bias=False
                )
        self.channels *= 2
        self.bn3 = torch.nn.BatchNorm1d(self.channels)
        self.pool3 = torch.nn.AvgPool1d(pool_size)
        self.dim //= pool_size

        # Sets the fourth convolutional layer
        self.conv4 = torch.nn.Conv1d(
                in_channels=self.channels,
                out_channels=self.channels * 2,
                kernel_size=kernel_size,
                padding=(kernel_size-1)//2,
                bias=False
                )
        self.channels *= 2
        self.bn4 = torch.nn.BatchNorm1d(self.channels)
        self.pool4 = torch.nn.AvgPool1d(pool_size)
        self.dim //= pool_size

        # Sets the fifth convolutional layer
        self.conv5 = torch.nn.Conv1d(
                in_channels=self.channels,
                out_channels=self.channels * 2,
                kernel_size=kernel_size,
                padding=(kernel_size-1)//2,
                bias=False
                )
        self.channels *= 2
        self.bn5 = torch.nn.BatchNorm1d(self.channels)
        self.pool5 = torch.nn.AvgPool1d(pool_size)
        self.dim //= pool_size

        # Sets the sixth convolutional layer
        self.conv6 = torch.nn.Conv1d(
                in_channels=self.channels,
                out_channels=self.channels * 2,
                kernel_size=kernel_size,
                padding=(kernel_size-1)//2,
                bias=False
                )
        self.channels *= 2
        self.bn6 = torch.nn.BatchNorm1d(self.channels)
        self.pool6 = torch.nn.AvgPool1d(pool_size)
        self.dim //= pool_size

        # Sets the seventh convolutional layer
        self.conv7 = torch.nn.Conv1d(
                in_channels=self.channels,
                out_channels=self.channels * 2,
                kernel_size=kernel_size,
                padding=(kernel_size-1)//2,
                bias=False
                )
        self.channels *= 2
        self.bn7 = torch.nn.BatchNorm1d(self.channels)
        self.pool7 = torch.nn.AvgPool1d(pool_size)
        self.dim //= pool_size

        # Sets the branch output layer
        flat_dims = self.channels * self.dim
        self.output = torch.nn.Linear(flat_dims, 1 << n_bits)

    def forward(self, x):
        """
        Inputs: torch.Tensor of shape (batch_size, 1, dim)
        Outputs: torch.Tensor of shape (batch_size, 1 << n_bits)
        """
        x = x[:, None, :]
        # First convolutional layer
        x_ = self.conv1(x)
        x_ = self.bn1(x_)
        x_ = torch.nn.functional.relu(x_)
        x_ = self.pool1(x_)

        # Second convolutional layer
        x_ = self.conv2(x_)
        x_ = self.bn2(x_)
        x_ = torch.nn.functional.relu(x_)
        x_ = self.pool2(x_)

        # Third convolutional layer
        x_ = self.conv3(x_)
        x_ = self.bn3(x_)
        x_ = torch.nn.functional.relu(x_)
        x_ = self.pool3(x_)

        # Fourth convolutional layer
        x_ = self.conv4(x_)
        x_ = self.bn4(x_)
        x_ = torch.nn.functional.relu(x_)
        x_ = self.pool4(x_)
        
        # Fifth convolutional layer
        x_ = self.conv5(x_)
        x_ = self.bn5(x_)
        x_ = torch.nn.functional.relu(x_)
        x_ = self.pool5(x_)

        # Sixth convolutional layer
        x_ = self.conv6(x_)
        x_ = self.bn6(x_)
        x_ = torch.nn.functional.relu(x_)
        x_ = self.pool6(x_)

        # Seventh convolutional layer
        x_ = self.conv7(x_)
        x_ = self.bn7(x_)
        x_ = torch.nn.functional.relu(x_)
        x_ = self.pool7(x_)

        # Flattens the feature maps
        x_flat = x_.view(x_.size()[0], -1)
        y_ = self.output(x_flat)
        return y_

# class WoutersBranch(torch.nn.Module):
#     """
#     Branch instantiating Wouters et al.'s architecture.
#     """
#     def __init__(
#             self,
#             dim_input: int,
#             n_bits: int = 8
#             ):
#         super(WoutersBranch, self).__init__()
#         # Sets the dimensionalities of the data (..., channels, dim)
#         self.channels = 1
#         self.dim = dim_input
# 
#         # Pre-processing
#         # self.preprocess = transforms.Normalize((0,), (1,))
#         self.pool0 = torch.nn.AvgPool1d(kernel_size=2, stride=2)
#         self.dim //= 2
# 
#         # 2nd convolutional block
#         self.conv2 = torch.nn.Conv1d(
#                 in_channels=1,
#                 out_channels=64,
#                 kernel_size=15,
#                 padding=(15-1)//2
#                 )
#         torch.nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
#         self.bn2 = torch.nn.BatchNorm1d(num_features=64)
#         self.pool2 = torch.nn.AvgPool1d(kernel_size=50, stride=50)
#         self.dim //= 50
# 
#         # 3rd convolutional block
#         self.conv3 = torch.nn.Conv1d(
#                 in_channels=64,
#                 out_channels=128,
#                 kernel_size=3,
#                 padding=(3-1)//2
#                 )
#         torch.nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
#         self.bn3 = torch.nn.BatchNorm1d(num_features=128)
#         self.pool3 = torch.nn.AvgPool1d(kernel_size=2, stride=2)
#         self.dim //= 2
#         self.channels = 128
# 
#         # Sets the branch output layer
#         flat_dims = self.channels * self.dim
# 
#         # Classification part
#         self.dense1 = torch.nn.Linear(flat_dims, 20)
#         torch.nn.init.kaiming_uniform_(self.dense1.weight, mode='fan_in', nonlinearity='selu')
#         # self.bndense1 = torch.nn.BatchNorm1d(num_features=20)
# 
#         self.dense2 = torch.nn.Linear(20, 20)
#         torch.nn.init.kaiming_uniform_(self.dense2.weight, mode='fan_in', nonlinearity='selu')
#         # self.bndense2 = torch.nn.BatchNorm1d(num_features=20)
# 
#         self.dense3 = torch.nn.Linear(20, 20)
#         torch.nn.init.kaiming_uniform_(self.dense3.weight, mode='fan_in', nonlinearity='selu')
#         # self.bndense3 = torch.nn.BatchNorm1d(num_features=20)
#         
#         # Logit layer
#         self.output = torch.nn.Linear(20, 1 << n_bits)
# 
# 
#     def forward(self, x):
#         """
#         WARNING: selu and BN are switched compared to Wouters et al.
#         """
#         # # Pre-processing
#         x = x[:, None, :]
#         # x = self.preprocess(x)
#         x = self.pool0(x)
# 
#         # 2nd convolutional block
#         x = self.conv2(x)
#         x = torch.nn.functional.selu(x)
#         x = self.bn2(x)
#         x = self.pool2(x)
# 
#         # 3rd convolutional block
#         x = self.conv3(x)
#         x = torch.nn.functional.selu(x)
#         x = self.bn3(x)
#         x = self.pool3(x)
# 
#         # Flatten
#         x = x.view(x.size()[0], -1)
# 
#         # Classification part
#         x = self.dense1(x)
#         # x = self.bndense1(x)
#         x = torch.nn.functional.selu(x)
# 
#         x = self.dense2(x)
#         # x = self.bndense2(x)
#         x = torch.nn.functional.selu(x)
# 
#         x = self.dense2(x)
#         # x = self.bndense2(x)
#         x = torch.nn.functional.selu(x)
# 
#         # Logits layer
#         x = self.output(x)
#         return x

        

class ASCADRecombine(Recombine):
    def __init__(
            self,
            keys_leakages: List[str],
            n_bits: int = 8,
            normalize_probs: bool = False
            ):
        super(ASCADRecombine, self).__init__(keys_leakages, n_bits, normalize_probs)
        self.tol = 1e-5

    def __repr__(self):
        return super().__repr__() + " with {}".format(convolve_affine)

    def forward(self, leakages: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Computes the log_probas for all the shares
        log_probas = {
                k: torch.nn.functional.log_softmax(val, dim=-1) 
                for k, val in leakages.items()
                }
        
        # Softmax normalization before convolution if specified
        if self.normalize_probs:
            leakages = {k: self.normalize(val) for k, val in leakages.items()}

        # Splits the distros
        p_alpha = torch.stack([val for k, val in leakages.items() if "alpha" in k], dim=1)
        p_beta = torch.stack([val for k, val in leakages.items() if "beta" in k], dim=1)
        p_masked = {k: val for k, val in leakages.items() if "masked" in k}

        # Applies the discrete convolution to recombine from the shares
        for masked_key, target in p_masked.items():
            res = convolve_affine(p_alpha, target, p_beta)
            if self.normalize_probs:
                res = torch.clip(res, min=self.tol, max=1)
                res = torch.log(res)
            else:
                res = torch.nn.functional.log_softmax(res, dim=-1)
            # Sets the right name for the target key
            target_key = "target_" + masked_key.split("_")[1]
            log_probas[target_key] = res

        return log_probas

class ASCADNet(torch.nn.Module):
    """
    A model to infer over the ASCAD dataset.
    """
    def __init__(self,
                 leakage_dims: Dict[str, int],
                 normalize_probs: bool = False,
                 **kwargs):
        """
        Params:
            * shares_dims: dict of dimensionality for each leakage.
            * scheme: type of masking scheme (boolean, arithmetical, etc)
        """
        super(ASCADNet, self).__init__()
        self.leakage_dims = leakage_dims
        self.normalize_probs = normalize_probs

        # Instantiates the branches
        branches_ = {
                k: ASCADBranch(
                    dim_input=val, 
                    n_filters_1=2, 
                    kernel_size=3, 
                    pool_size=2, 
                    n_bits=8
                    )
                # k: WoutersBranch(dim_input=val,n_bits=8)
                for k, val in self.leakage_dims.items()
                }
        self.branches = torch.nn.ModuleDict(branches_)
        
        # Instantiates the masking encoding
        self.recombine = GroupRecombine(list(self.leakage_dims.keys()), n_bits=8, normalize_probs=self.normalize_probs, scheme="boolean")

    def forward(self, leak_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Forward pass on each 
        branch_res = {k: self.branches[k](val) for k, val in leak_dict.items()}
        # Forward pass for recombination
        return self.recombine(branch_res)

# class BBWouters(torch.nn.Module):
#     """
#     A class of Models that implements 
#     """
#     def __init__(
#             self,
#             leakage_dims: Dict[str, int],
#             n_bits: int = 8,
#             ):
#         super(BBWouters, self).__init__()
#         self.leakage_dim = sum(leakage_dims.values())
#         self.n_bits = n_bits
# 
#         # Splits the distros
#         dim_masked = {k: val for k, val in leakage_dims.items() if "masked" in k}
# 
#         self.branches_ = {
#                 k: WoutersBranch(
#                     dim_input=self.leakage_dim, 
#                     n_bits=self.n_bits, 
#                     )
#                 for k, val in dim_masked.items()
#                 }
#         self.branches = torch.nn.ModuleDict(self.branches_)
# 
#     def forward(self, leak_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """
#         Forward pass for the BB model.
#         """
#         stacked_leakage = torch.hstack([val for k, val in leak_dict.items()])
#         log_probas = dict()
#         for k, val in self.branches.items():
#             tmp = val(stacked_leakage)
#             target_key = "target_" + k.split("_")[1]
#             log_probas[target_key] = torch.nn.functional.log_softmax(tmp, dim=-1)
#         return log_probas
