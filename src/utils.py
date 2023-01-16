"""
Utilitary script
Author: Loïc Masure (loic.masure@uclouvain.be)
Date: 21/04/2021
"""
from typing import List, Tuple, Callable, Dict, Optional, Union
import os
import logging
import random
import numpy as np
import torch


#AES Sbox
Sbox = np.array([ 0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01,
    0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76, 0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59,
    0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0, 0xB7, 0xFD,
    0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8,
    0x31, 0x15, 0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12,
    0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75, 0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E,
    0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84, 0x53, 0xD1,
    0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C,
    0x58, 0xCF, 0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9,
    0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8, 0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D,
    0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2, 0xCD, 0x0C,
    0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D,
    0x19, 0x73, 0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE,
    0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB, 0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06,
    0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79, 0xE7, 0xC8,
    0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A,
    0xAE, 0x08, 0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD,
    0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A, 0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03,
    0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E, 0xE1, 0xF8,
    0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55,
    0x28, 0xDF, 0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99,
    0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16 ])
# ################### Log/Alog tables with 3 as generator #####################
# Polynomial in GF(2^3): 0x11
alog_3 = [1, 3, 5, 4, 7, 2, 6, 1]
log_3 = [-1, 0, 5, 1, 3, 2, 6, 4]

# Polynomial in GF(2^4): 0x13
alog_4 = [1, 3, 5, 15, 2, 6, 10, 13, 4, 12, 7, 9, 8, 11, 14, 1]
log_4 = [-1, 0, 4, 1, 8, 2, 5, 10, 12, 11, 6, 13, 9, 7, 14, 3]

# Polynomial in GF(2^6): 0x49
alog_6 = [1, 3, 5, 15, 17, 51, 28, 36, 37, 38, 35, 44, 61, 14, 18, 54, 19, 53,
          22, 58, 7, 9, 27, 45, 62, 11, 29, 39, 32, 41, 50, 31, 33, 42, 55, 16,
          48, 25, 43, 52, 21, 63, 8, 24, 40, 49, 26, 46, 59, 4, 12, 20, 60, 13,
          23, 57, 2, 6, 10, 30, 34, 47, 56, 1]
log_6 = [-1, 0, 56, 1, 49, 2, 57, 20, 42, 21, 58, 25, 50, 53, 13, 3, 35, 4, 14,
         16, 51, 40, 18, 54, 43, 37, 46, 22, 6, 26, 59, 31, 28, 32, 60, 10, 7,
         8, 9, 27, 44, 29, 33, 38, 11, 23, 47, 61, 36, 45, 30, 5, 39, 17, 15,
         34, 62, 55, 19, 48, 52, 12, 24, 41]

# Polynomial in GF(2^7): 0x83
alog_7 = [1, 3, 5, 15, 17, 51, 85, 124, 7, 9, 27, 45, 119, 26, 46, 114, 21, 63,
          65, 64, 67, 70, 73, 88, 107, 62, 66, 69, 76, 87, 122, 13, 23, 57, 75,
          94, 97, 32, 96, 35, 101, 44, 116, 31, 33, 99, 38, 106, 61, 71, 74,
          93, 100, 47, 113, 16, 48, 80, 115, 22, 58, 78, 81, 112, 19, 53, 95,
          98, 37, 111, 50, 86, 121, 8, 24, 40, 120, 11, 29, 39, 105, 56, 72,
          91, 110, 49, 83, 118, 25, 43, 125, 4, 12, 20, 60, 68, 79, 82, 117,
          28, 36, 108, 55, 89, 104, 59, 77, 84, 127, 2, 6, 10, 30, 34, 102, 41,
          123, 14, 18, 54, 90, 109, 52, 92, 103, 42, 126, 1]
log_7 = [-1, 0, 109, 1, 91, 2, 110, 8, 73, 9, 111, 77, 92, 31, 117, 3, 55, 4,
         118, 64, 93, 16, 59, 32, 74, 88, 13, 10, 99, 78, 112, 43, 37, 44, 113,
         39, 100, 68, 46, 79, 75, 115, 125, 89, 41, 11, 14, 53, 56, 85, 70, 5,
         122, 65, 119, 102, 81, 33, 60, 105, 94, 48, 25, 17, 19, 18, 26, 20,
         95, 27, 21, 49, 82, 22, 50, 34, 28, 106, 61, 96, 57, 62, 97, 86, 107,
         6, 71, 29, 23, 103, 120, 83, 123, 51, 35, 66, 38, 36, 67, 45, 52, 40,
         114, 124, 104, 80, 47, 24, 101, 121, 84, 69, 63, 54, 15, 58, 42, 98,
         87, 12, 76, 72, 30, 116, 7, 90, 126, 108]

# Polynomial in GF(2^8): same as Rijndael
log_8 = [-1, 0, 25, 1, 50, 2, 26, 198, 75, 199, 27, 104, 51, 238, 223, 3, 100,
         4, 224, 14, 52, 141, 129, 239, 76, 113, 8, 200, 248, 105, 28, 193,
         125, 194, 29, 181, 249, 185, 39, 106, 77, 228, 166, 114, 154, 201, 9,
         120, 101, 47, 138, 5, 33, 15, 225, 36, 18, 240, 130, 69, 53, 147, 218,
         142, 150, 143, 219, 189, 54, 208, 206, 148, 19, 92, 210, 241, 64, 70,
         131, 56, 102, 221, 253, 48, 191, 6, 139, 98, 179, 37, 226, 152, 34,
         136, 145, 16, 126, 110, 72, 195, 163, 182, 30, 66, 58, 107, 40, 84,
         250, 133, 61, 186, 43, 121, 10, 21, 155, 159, 94, 202, 78, 212, 172,
         229, 243, 115, 167, 87, 175, 88, 168, 80, 244, 234, 214, 116, 79, 174,
         233, 213, 231, 230, 173, 232, 44, 215, 117, 122, 235, 22, 11, 245, 89,
         203, 95, 176, 156, 169, 81, 160, 127, 12, 246, 111, 23, 196, 73, 236,
         216, 67, 31, 45, 164, 118, 123, 183, 204, 187, 62, 90, 251, 96, 177,
         134, 59, 82, 161, 108, 170, 85, 41, 157, 151, 178, 135, 144, 97, 190,
         220, 252, 188, 149, 207, 205, 55, 63, 91, 209, 83, 57, 132, 60, 65,
         162, 109, 71, 20, 42, 158, 93, 86, 242, 211, 171, 68, 17, 146, 217,
         35, 32, 46, 137, 180, 124, 184, 38, 119, 153, 227, 165, 103, 74, 237,
         222, 197, 49, 254, 24, 13, 99, 140, 128, 192, 247, 112, 7]
alog_8 = [1, 3, 5, 15, 17, 51, 85, 255, 26, 46, 114, 150, 161, 248, 19, 53, 95,
          225, 56, 72, 216, 115, 149, 164, 247, 2, 6, 10, 30, 34, 102, 170,
          229, 52, 92, 228, 55, 89, 235, 38, 106, 190, 217, 112, 144, 171, 230,
          49, 83, 245, 4, 12, 20, 60, 68, 204, 79, 209, 104, 184, 211, 110,
          178, 205, 76, 212, 103, 169, 224, 59, 77, 215, 98, 166, 241, 8, 24,
          40, 120, 136, 131, 158, 185, 208, 107, 189, 220, 127, 129, 152, 179,
          206, 73, 219, 118, 154, 181, 196, 87, 249, 16, 48, 80, 240, 11, 29,
          39, 105, 187, 214, 97, 163, 254, 25, 43, 125, 135, 146, 173, 236, 47,
          113, 147, 174, 233, 32, 96, 160, 251, 22, 58, 78, 210, 109, 183, 194,
          93, 231, 50, 86, 250, 21, 63, 65, 195, 94, 226, 61, 71, 201, 64, 192,
          91, 237, 44, 116, 156, 191, 218, 117, 159, 186, 213, 100, 172, 239,
          42, 126, 130, 157, 188, 223, 122, 142, 137, 128, 155, 182, 193, 88,
          232, 35, 101, 175, 234, 37, 111, 177, 200, 67, 197, 84, 252, 31, 33,
          99, 165, 244, 7, 9, 27, 45, 119, 153, 176, 203, 70, 202, 69, 207, 74,
          222, 121, 139, 134, 145, 168, 227, 62, 66, 198, 81, 243, 14, 18, 54,
          90, 238, 41, 123, 141, 140, 143, 138, 133, 148, 167, 242, 13, 23, 57,
          75, 221, 124, 132, 151, 162, 253, 28, 36, 108, 180, 199, 82, 246, 1]
############################ Log/ALog tables for prime groups #################
log_13 = [-1, 12, 1, 4, 2, 9, 5, 11, 3, 8, 10, 7, 6]
alog_13 = [1, 2, 4, 8, 3, 6, 12, 11, 9, 5, 10, 7, 1]


def get_log_tables(n_bits: int) -> Tuple[List, List]:
    """
    Returns the corresponding log/alog tables.
    """
    if n_bits == 3:
        return log_3, alog_3
    elif n_bits == 4:
        return log_4, alog_4
    elif n_bits == 6:
        return log_6, alog_6
    elif n_bits == 7:
        return log_7, alog_7
    elif n_bits == 8:
        return log_8, alog_8
    else:
        msg = "The log tables do not exist for {} bits".format(n_bits)
        raise NotImplementedError(msg)


def is_power_of_two(x) -> bool:
    log2 = np.log2(x)
    return log2 == np.floor(log2)


def multGF(a, b, log, alog):
    """
    Field multiplication a x b
    """
    if (a == 0) or (b == 0):
        return 0
    else:
        return alog[(log[a]+log[b]) % (len(log) - 1)]


def divGF(a, b, log, alog):
    """
    Field division a / b
    """
    if (a == 0) or (b == 0):
        return 0
    else:
        modulo = len(log) - 1
        return alog[(log[a]-log[b]) % modulo]


def group_law(op: str, length: int) -> Callable:
    """
    Defines the *inverse* of the inner law of the group.
    """
    if op == "boolean":
        return lambda x, y: x ^ y
    elif op == "arithmetical":
        return lambda x, y: (x - y) % length
    elif op == "multiplicative":
        n_bits = int(np.log2(length))
        log, alog = get_log_tables(n_bits)
        return np.vectorize(lambda a, b: divGF(a, b, log, alog))
    else:
        msg = "The '{}' law not recognized (yet).".format(op)
        raise NotImplementedError(msg)


def isProbDist(x: torch.Tensor, axis: int = -1) -> bool:
    """
    Checks whether the tensor is a probability distribution along the chosen
    dimension.
    """
    # Checks > 0
    all_positive = (x >= 0)
    if not all_positive.all():
        idx_pb = np.where(all_positive is False)
        tmp = x[idx_pb]
        msg = "Probabilities are not all non negative: see at entries"
        msg += " {} with value {}".format(idx_pb, tmp.item())
        raise ValueError(msg)

    # Checks sums to 1
    one = torch.tensor(1, dtype=torch.float)
    sums_to_1 = torch.isclose(x.sum(dim=axis), one).all()
    if not sums_to_1:
        diff = x.sum(dim=axis) - one
        msg = "The sum of probabilities does not sum to 1: errors range in "
        msg += "[{} : {}]".format(diff.min(), diff.max())
        raise ValueError(msg)
    return bool(all_positive.all()) and bool(sums_to_1)


def set_logger(log_name, log_path):
    """
    Sets the logger to log info in terminal and file 'log_path'.
    Params:
        * log_path: (str) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(os.path.join(log_path, log_name))
        log_txt = "%(asctime)s:%(levelname)s: %(message)s"
        file_handler.setFormatter(logging.Formatter(log_txt))
        logger.addHandler(file_handler)

        # Logging to a console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    return logger


def set_deterministic(seed: int, num_threads: int):
    """
    Sets the environment to be deterministic, across platforms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "1.7." in torch.__version__:
        torch.set_deterministic(True)
    elif "1.8." in torch.__version__:
        torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.set_num_threads(num_threads)


# ########################### Fast DataLoader #################################
class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: 
    https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/
    """
    def __init__(self, *dicts, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *dicts: list of dict of tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        key_0 = next(iter(dicts[0]))
        self.dataset_len = dicts[0][key_0].shape[0]
        for dico in dicts:
            for k, tensor in dico.items():
                assert(tensor.shape[0] == self.dataset_len)
        # assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.dicts = dicts

        # self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            for dico in self.dicts:
                dico = {k: tensor[r] for k, tensor in dico.items()}
            #self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        
        batch = tuple({k: val[self.i:self.i+self.batch_size] for k, val in dico.items()} for dico in self.dicts)

        #batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
###############################################################################


def forward_pass(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 model: torch.nn.Module,
                 criterion: torch.nn.Module
                 ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Applies the forward pass to the batch of data through the model
    """
    # Loads the batch
    traces, labels, shares = batch
    # Computes the model output and loss
    outputs = model(traces)
    loss_targets = {k: criterion(val, labels[k]) for k, val in outputs.items() if "target" in k}
    loss_shares = {k: criterion(val, shares[k]) for k, val in outputs.items() if "target" not in k}
    return loss_targets, loss_shares


def backward_pass(
        optimizer: torch.optim.Optimizer, 
        loss: torch.Tensor, 
        scheduler: torch.optim.lr_scheduler._LRScheduler = None
        ):
    """
    Applies the backward pass to the model.
    Computes gradient and steps the optimizer
    """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()


def epoch_loop(model: torch.nn.Module,
               loader: FastTensorDataLoader,
               criterion: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler._LRScheduler = None,
               train: bool = True,
               white_box: bool = False
               ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Performs the training loop
    """
    loss_targets_hist = []
    loss_shares_hist = []

    if train:
        model.train()
    else:
        model.eval()

    for batch in loader:
        loss_targets, loss_shares = forward_pass(batch, model, criterion)
        # Averages for backward
        loss_mean = torch.mean(torch.stack([val for val in loss_targets.values()]))
        if train:
            if not white_box:
                backward_pass(optimizer, loss_mean, scheduler)
            else:
                loss_shares_mean = torch.mean(torch.stack([val for val in loss_shares.values()]))
                backward_pass(optimizer, loss_shares_mean, scheduler)
        # Records for history
        loss_targets = {k: val.item() / np.log(2) for k, val in loss_targets.items()}
        loss_shares = {k: val.item() / np.log(2) for k, val in loss_shares.items()}
        loss_targets_hist.append(loss_targets)
        loss_shares_hist.append(loss_shares)

    # Averages over the history
    loss_targets_ = {k: np.mean([l[k] for l in loss_targets_hist]) for k in loss_targets_hist[0].keys()}
    loss_shares_ = {k: np.mean([l[k] for l in loss_shares_hist]) for k in loss_shares_hist[0].keys()}
    return loss_targets_, loss_shares_
