"""
Module for generating simulated traces.
Author: Loïc Masure
Date: 21/04/2021
"""
from typing import Callable, Optional, Tuple, Iterable, Dict
import torch
import numpy as np
from numpy.fft import fft, ifft
import scipy.stats as stats
import copy
import warnings
from torch.utils.data import Dataset

import utils

# To have a device-agnostic software
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ####################### Different leakage models ############################
def lsb(target: int) -> int:
    """
    Returns the LSB of the input target of type int.
    """
    return target & 0b1

def hw(target: int) -> int:
    """
    Returns the Hamming weight of the input target of type int.
    """
    return bin(target).count("1")


def leak_id(target: np.ndarray) -> np.ndarray:
    return target #| 0x1  # & 0x01

hw_vec = np.vectorize(hw)
lsb_vec = np.vectorize(lsb)

def select_leakage_model(leak_model: str = "hw") -> Callable:
    """
    Maps the leakage model passed in arguments to the corresponding callable
    leakage model defined in this module.
    """
    if leak_model == "hw":
        return hw_vec
    elif leak_model == "lsb":
        return lsb_vec
    elif leak_model == "leak_id":
        return leak_id
    else:
        raise ValueError("The '{}' leakage model is not supported yet.".format(leak_model))

# ########################## Abstract class for Simulation ####################
class Simulation(Dataset):
    """
    A class managing the simulated dataset.
    An instance of this class is used to generate a data loader,
    in charge of correctly loading the training and validation data
    inside the training loop.
    """
    def __init__(self,
                 n_samples: int,
                 sigma: float,
                 order: int,
                 n_bits: int,
                 n_targets: int = 1,
                 leak_hard: Callable = hw_vec):
        self.n_samples = n_samples
        self.sigma = sigma
        self.n_shares = order+1
        self.n_targets = n_targets
        self.n_bits = n_bits
        self.leak_hard = leak_hard
        self.set_leak_models()
        self.refresh_data()

    def refresh_data(self):
        """
        Renews the dataset. Always called when instantiating, or when emulating
        a dataset of infinite size.
        """
        self.labels, self.shares = self.draw_shares()
        self.leakages = self.draw_leakages()

        # Converts to torch.Tensor
        self.leakages = {k: torch.tensor(val, dtype=torch.float, device=device) for k, val in self.leakages.items()}
        self.labels = {k: torch.tensor(val, dtype=torch.long, device=device) for k, val in self.labels.items()}
        self.shares = {k: torch.tensor(val, dtype=torch.long, device=device) for k, val in self.shares.items()}

    def draw_shares(self) -> Tuple[Dict, Dict]:
        raise NotImplementedError

    def set_leak_models(self):
        raise NotImplementedError

    def draw_leakages(self) -> Dict:
        """
        Applies the leakage models
        """
        leakages = {k: self.leak_models[k](val) for k, val in
                self.shares.items()}
        noise = {k: np.random.randn(*val.shape) * self.sigma for k, val in
                leakages.items()}
        return {k: val + noise[k] for k, val in leakages.items()}

    @staticmethod
    def pdf_x_share(x: np.ndarray,
            sigma: float,
            hyp_set: Iterable,
            leak_model: Callable
            ) -> np.ndarray:
        """
        Computes the pdf vector P(X = x | Z = s) for each hypothetical discrete
        value s in hyp_set. X here is assumed to be multivariate.
        Params:
            * x: array of shape (n_draws, dim_leak) the draws from which the pdf
            must be computed.
            * sigma: the noise parameter of the Gaussian law.
            * hyp_set: the discrete set of hypothetical value s may take.
        Return:
            * phis: array of shape (n_draws, dim_leak, len(hyp_set))
        """
        phis_transp = [stats.norm(loc=leak_model(s), scale=sigma).pdf(x.cpu()) for s in hyp_set]
        phis = np.array(phis_transp).transpose(1, 2, 0)
        return phis.prod(axis=1)  # Assumes independence of elem. leakages

    def pdf_x_s(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def pdf_s_x(p_x_s: np.ndarray) -> np.ndarray:
        """
        Applies Bayes' theorem to compute P(Z = s | X = x) w.r.t.
        P(X = X | Z = s).
        Assumes that the marginal pdf on Z, P(Z = s), is uniform.
        Params:
            * p_x_s: vector of shape (n_draws, n_shares, #Z)
        Returns:
            * pdf_s_x: vector of shape (n_draws, n_shares, #Z)
        """
        sum = p_x_s.sum(axis=-1, keepdims=True)
        return p_x_s / sum

    @staticmethod
    def compute_entropy(pdf_s_x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Given a vector of conditional pdfs P(Z = s | X = x), estimates the
        conditional entropy H(Z | X).
        Params:
            * pdf_s_x: vector of shape (*x.shape, #Z). Should be (n_draws, 256)
        Return:
            * h: the conditional entropy H(Z | X = x). Shape: pdf_s_x.shape[1:]
            * h_tot: the conditional entropy h, averaged over X. Scalar.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmp2 = pdf_s_x * np.log2(pdf_s_x)
        tmp2[np.isnan(tmp2)] = 0  # x*log(x) -> 0 when x -> 0
        h = -np.sum(tmp2, axis=-1)
        return h, h.mean()

    def entropy(self):
        """
        Returns the estimated entropy of the dataset.
        """
        # Computes the pdfs for each target
        toto = self.pdf_x_s()
        # Normalizes with Bayes
        p_s_given_x = {k: self.pdf_s_x(val) for k, val in toto.items()}
        # Computes the entropy for each target
        entrop = {k: self.compute_entropy(val)[1] for k, val in p_s_given_x.items()}
        return entrop, np.mean(list(entrop.values()))

    @staticmethod
    def compute_en(pdf_s_x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Given a vector of conditional pdfs P(Y=s | L=l), estimates the
        Euclidean distance EN(Y;L).
        Params:
            * pdf_s_x: vector of shape (*x.shape, #Y). Should be (n_draws, 256)
        Return:
            * sd:       the statistical distance computed for each draw.
                        Shape: pdf_s_x.shape[1:]
            * sd_tot:   the statistical distance, averaged over L. Scalar.
        """
        p_uniform = 1/pdf_s_x.shape[-1]
        distance_to_uniform = np.sqrt(((pdf_s_x - p_uniform)**2).sum(axis=-1))
        return distance_to_uniform, distance_to_uniform.mean()

    def euclidean_distance(self):
        """
        Returns the Euclidean distance to the uniform distribution.
        """
        # Computes the pdfs for each target
        toto = self.pdf_x_s()
        # Normalizes with Bayes
        p_s_given_x = {k: self.pdf_s_x(val) for k, val in toto.items()}
        # Computes the statistical distance for each target
        ed = {k: self.compute_en(val)[1] for k, val in p_s_given_x.items()}
        return ed, np.mean(list(ed.values()))

    @staticmethod
    def compute_sd(pdf_s_x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Given a vector of conditional pdfs P(Y=s | L=l), estimates the
        statistical distance SD(Y;L).
        Params:
            * pdf_s_x: vector of shape (*x.shape, #Y). Should be (n_draws, 256)
        Return:
            * sd:       the statistical distance computed for each draw.
                        Shape: pdf_s_x.shape[1:]
            * sd_tot:   the statistical distance, averaged over L. Scalar.
        """
        p_uniform = 1/pdf_s_x.shape[-1]
        distance_to_uniform = np.abs(pdf_s_x - p_uniform).sum(axis=-1) / 2
        return distance_to_uniform, distance_to_uniform.mean()

    def stat_distance(self):
        """
        Returns the statistical distance to the uniform distribution.
        """
        # Computes the pdfs for each target
        toto = self.pdf_x_s()
        # Normalizes with Bayes
        p_s_given_x = {k: self.pdf_s_x(val) for k, val in toto.items()}
        # Computes the statistical distance for each target
        sd = {k: self.compute_sd(val)[1] for k, val in p_s_given_x.items()}
        return sd, np.mean(list(sd.values()))

    def get_shapes(self):
        return {k: self.leakages[k].shape[1] for k in self.leakages.keys()}

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        traces = {k: val[idx] for k, val in self.leakages.items()}
        labels = {k: val[idx] for k, val in self.labels.items()}
        shares = {k: val[idx] for k, val in self.shares.items()}
        return traces, labels, shares
###############################################################################
class Group(Simulation):
    """
    A simulation class for higher-order masking w.r.t. group operation.
    """
    def __init__(self,
                 n_samples: int,
                 sigma: float,
                 order: int,
                 n_bits: int,
                 n_targets: int,
                 group_op: Callable,
                 leak_hard: Callable = hw_vec):
        self.group_op = group_op
        if group_op == "boolean":
            self.reduce = self.reduce_bool
            self.conv = conv_xor
        elif group_op == "arithmetical":
            self.reduce = self.reduce_arith
            self.conv = convolve_fft
        elif group_op == "multiplicative":
            self.reduce = self.reduce_mult
            self.conv = convolve_mult
        else:
            raise NotImplementedError("The '{}' scheme is not supported (yet)".format(self.group_op))
        super(Group, self).__init__(n_samples, sigma, order, n_bits, n_targets, leak_hard)

    @staticmethod
    def reduce_mult(x, n_bits):
        # Computation through log tables
        log, alog = utils.get_log_tables(n_bits)
        mult = np.vectorize(lambda a, b: utils.multGF(a, b, log, alog))
        s, ss = x[:, 0], x[:, 1:]
        for i in range(ss.shape[1]):
            s = mult(s, ss[:, i])
        return s

    @staticmethod
    def reduce_arith(x, n_bits):
        return x.sum(axis=1) % (1 << n_bits)
        # Comment out for particular case (prime fields)
        # return x.sum(axis=1) % ((1 << n_bits) - 1)

    @staticmethod
    def reduce_bool(x, n_bits):
        return np.bitwise_xor.reduce(x, axis=1)

    def draw_shares(self) -> Tuple[Dict, Dict]:
        # Set the range of the shares
        min_val = 0
        max_val = 1 << self.n_bits

        # Draws the masks
        shares = {"beta_{}".format(j): np.random.randint(min_val, max_val,
            (self.n_samples,)) for j in range(self.n_shares-1)}
        masks_array_ = np.array(list(shares.values())).transpose()

        # Draws the masked data and computes the secret
        labels = dict()
        for j in range(self.n_targets):
            shares["masked_{}".format(j)] = np.random.randint(min_val, max_val,
                    (self.n_samples,))
            if self.n_shares > 1:
                shares_array_ = np.concatenate((masks_array_,
                    shares["masked_{}".format(j)][:, None]), axis=1)
                secret = self.reduce(shares_array_, self.n_bits)
                labels["target_{}".format(j)] = secret
            else:
                labels["target_{}".format(j)] = shares["masked_{}".format(j)]

        return labels, shares

    def set_leak_models(self):
        """
        Sets the leakage models for each share individually.
        """
        shares_keys = []
        for i in range(self.n_shares-1):
            shares_keys.append("beta_{}".format(i))
        for j in range(self.n_targets):
            shares_keys.append("masked_{}".format(j))
        self.leak_models = {k: lambda x:
                self.leak_hard(np.array([x]).transpose()) for k in shares_keys}


    def pdf_x_s(self):
        min_val = 0
        max_val = 1 << self.n_bits
        phis = {k: self.pdf_x_share(val, self.sigma, hyp_set=range(min_val,
            max_val), leak_model=self.leak_models[k]) for k, val in self.leakages.items()}


        # Splits the probs
        p_masked = {k: val for k, val in phis.items() if "masked" in k}
        # Case without masking
        if self.n_shares == 1:
            return p_masked

        # Case with masking
        p_beta = np.array([val for k, val in phis.items() if "beta" in k]).transpose(1, 0, 2)
        res = dict()
        for k, masked in p_masked.items():
            # Computes the convolution w.r.t. group inner law
            phis_op = np.concatenate((p_beta, masked[:, None, :]), axis=1)
            res[k] = self.conv(phis_op)

        return res


# ################################ Fast Transforms ############################
def fwht(a: np.ndarray):
    """
    In-place Fast Walsh-Hadamard Transform of array a along the *first* axis.
    """
    h = 1
    length = a.shape[0]
    while h < length:
        for i in range(0, length, h * 2):
            for j in range(i, i + h):
                x = copy.deepcopy(a[j])
                y = copy.deepcopy(a[j+h])
                a[j] = x + y
                a[j+h] = x - y
        h *= 2


def conv_xor(phis: np.ndarray) -> np.ndarray:
    """
    Applies the convolution according to the xor, with FWHT.
    """
    # Computes the Walsh-Hadamard transform.
    fwht(phis.transpose(2, 0, 1))
    # Multiplication in the transform space is equivalent to convolution.
    conv = phis.prod(axis=1)
    # Back to the initial domain
    fwht(conv.transpose())
    return conv


def convolve_fft(x):
    # Computes the fft
    fft_ = fft(x, axis=-1).prod(axis=1, keepdims=True)
    # Goes back from the frequency domain
    return ifft(fft_, axis=-1).real


def convolve_mult(x: np.ndarray) -> np.ndarray:
    """
    x: Tensor of shape (batch_size, n_shares, length)
    The (n_shares-1) first shares are multiplicative masks (strictly positive)
    The last share denotes the masked data.
    """
    n_bits = int(np.log2(x.shape[-1]))
    log, alog = utils.get_log_tables(n_bits)
    # log, alog = utils.log_13, utils.alog_13
    # Turns to log space
    x_alog = x[:, :, alog[1:]]
    # Convolution like with arithmetical scheme
    prod_alog = convolve_fft(x_alog)
    # Turns back from log space
    # prod = prod_alog[:, log[1:]]
    log_np = np.array(log)
    prod = prod_alog[:, :, log_np-2]  # Don't know why -2 ...
    # Adds the case where the masked data is 0, bc probs not normalized yet
    # Comment out when the secret value cannot be 0
    res_0 = np.zeros((prod.shape[0], prod.shape[1], 1))
    # Comment out when the secret value may be 0
    # res_0 = x[:, -1:, :1] * x[:, :-1, 1:].sum(axis=-1, keepdims=True).prod(axis=1, keepdims=True)
    return np.concatenate((res_0, prod[:, :, 1:]), axis=-1)
###############################################################################

# ######################## Test draws for ASCADv2-like leakages ###############
def compute_GTab(n_bits):
    """
    Prepares the software representation of the leakage.
    """
    mult = np.vectorize(lambda a, b: utils.multGF(a, b, log, alog))
    log, alog = utils.get_log_tables(n_bits)
    GTabs = np.array([[mult(x, y) for y in range(1, 1 << n_bits)] for x in range(1, 1 << n_bits)])
    def leak_alpha_(alpha):
        return GTabs[alpha-1]  # (n_draws, D_alphas)
        # GTab = np.array([mult(alpha, i) for i in range(1, 1 << n_bits)])
        # return GTab.transpose()  # (n_draws, D_alphas)
    return leak_alpha_

def convolve_affine(
        alphas: np.ndarray,
        masked: np.ndarray,
        betas: np.ndarray
        ) -> torch.Tensor:
    """
    Performs the convolution for the affine masking.
    NB : as it is implemented in the ASCAD paper, one should use cross-correlation
    instead of convolutions !
    Dimensions:
        * alphas, betas: [batch_size, (n_shares-1)//2, 2^n]
        * masked: [batch_size, 1, 2^n]
    """
    betas = np.concatenate((masked[:, None, :], betas), axis=1)
    betas = conv_xor(betas)

    # Turn to log space
    n_bits = int(np.log2(masked.shape[-1]))
    log, alog = utils.get_log_tables(n_bits)
    betas_alog = betas[:, alog[:-1]]
    alphas_alog = alphas[:, :, alog[:-1]]

    # Applies FFT
    betas_fft = np.fft.fft(betas_alog, axis=-1)
    alphas_fft = np.fft.fft(alphas_alog, axis=-1)
    alphas_fft = np.conjugate(alphas_fft)  # Important to take the conjugate here
    alphas_prod = alphas_fft.prod(axis=1)
    fft_prod = alphas_prod * betas_fft
    prod_alog = np.fft.ifft(fft_prod, axis=-1).real

    # Turn back from log space
    log_np = np.array(log[1:])
    prod = prod_alog[:, log_np]  # Don't know why -2 ...

    # Adds the case where the masked data is 0
    res_0 = betas[:, :1] * alphas[:, :, 1:].sum(axis=-1, keepdims=True).prod(axis=1)
    return np.concatenate((res_0, prod), axis=-1)

class ASCAD(Simulation):
    """
    A simulation class aiming at reproducing the ASCAD characters
    """
    def __init__(self,
                 n_samples: int,
                 sigma: float,
                 order: int,
                 n_bits: int,
                 n_targets: int,
                 leak_alpha_soft: Callable = None,
                 leak_beta_soft: Callable = None,
                 leak_hard: Callable = hw_vec):
        if leak_alpha_soft is None:
            self.leak_alpha_soft = compute_GTab(n_bits)
        else:
            self.leak_alpha_soft = leak_alpha_soft
        if leak_beta_soft is None:
            self.leak_beta_soft = lambda x: np.array([x ^ j for j in range(1 << n_bits)]).transpose()
        else:
            self.leak_beta_soft = leak_beta_soft
        super(ASCAD, self).__init__(n_samples, sigma, order, n_bits, n_targets, leak_hard)

    def set_leak_models(self):
        # Lists all the leakages to consider
        shares_keys = []
        for i in range((self.n_shares-1)//2):
            shares_keys.append("alpha_{}".format(i))
            shares_keys.append("beta_{}".format(i))
        for j in range(self.n_targets):
            shares_keys.append("masked_{}".format(j))

        # Sets the leakage models
        leak_models = dict()
        for k in shares_keys:
            if "alpha" in k:
                leak_models[k] = lambda x: self.leak_hard(self.leak_alpha_soft(x))
            # elif "beta" in k:
            #     leak_models[k] = lambda x: self.leak_hard(self.leak_beta_soft(x))
            # elif "masked" in k:
            #     leak_models[k] = lambda x: self.leak_hard(self.leak_beta_soft(x))
            else:
                leak_models[k] = lambda x: self.leak_hard(np.array([x]).transpose())
        self.leak_models = leak_models

    def draw_shares(self) -> Tuple[Dict, Dict]:
        """
        Draws leakages according in an ASCADv2-like way. More precisely:
            * for the multiplicative share alpha, all the multiples of alpha leak.
            This denotes the leakages from the pre-computation table.
            * For the Boolean share beta, one leak 2^n_bits leakages following the
            same model.
            * For the masked data: one draws n_targets different shares, one for
            each targeted byte.
        """
        # Sets the dimensionality of the leakages
        log, alog = utils.get_log_tables(self.n_bits)
        mult = np.vectorize(lambda a, b: utils.multGF(a, b, log, alog))
        div = np.vectorize(lambda a, b: utils.divGF(a, b, log, alog))

        # Draws the masks
        shares, labels = dict(), dict()
        for i in range((self.n_shares-1)//2):
            shares["alpha_{}".format(i)] = np.random.randint(1, 1 << self.n_bits,
                    (self.n_samples,))
            shares["beta_{}".format(i)] = np.random.randint(0, 1 << self.n_bits,
                    (self.n_samples,))

        # Draws the masked data and computes the secrets
        labels = dict()
        for j in range(self.n_targets):
            shares["masked_{}".format(j)] = np.random.randint(0,
                    1 << self.n_bits, (self.n_samples,))
            # Computes the secret
            secret_alpha = np.ones(shares[f"masked_{j}"].shape, dtype=shares[f"masked_{j}"].dtype)
            secret_beta = np.zeros(shares[f"masked_{j}"].shape, dtype=shares[f"masked_{j}"].dtype)
            for i in range((self.n_shares-1)//2):
                secret_alpha = mult(secret_alpha, shares["alpha_{}".format(i)])
                secret_beta ^= shares["beta_{}".format(i)]
            labels["target_{}".format(j)] = div(shares[f"masked_{j}"] ^ secret_beta, secret_alpha)

        return labels, shares

    def pdf_x_s(self):
        phis = {k: self.pdf_x_share(val, self.sigma, hyp_set=range(1 << self.n_bits), leak_model=self.leak_models[k]) for k, val in self.leakages.items()}

        # Splits the probs
        p_alpha = np.array([val for k, val in phis.items() if "alpha" in k]).transpose(1, 0, 2)
        p_beta = np.array([val for k, val in phis.items() if "beta" in k]).transpose(1, 0, 2)
        p_masked = {k: val for k, val in phis.items() if "masked" in k}

        res = dict()
        for k, masked in p_masked.items():
            res[k] = convolve_affine(p_alpha, masked, p_beta)

        return res
###############################################################################
