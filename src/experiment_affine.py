"""
First simulation of a Hamming weight + additive Gaussian noise to fit.
We compare a vanilla one-layer MLP with its equivalent made of two sub-networks
linked with a Walsh-Hadamard (WH) convolution.

Features to add:
        * Extend to affine masking
        * Improve a bit the logging, e.g., reporting every 10/100 epochs
        * Restrain this file to one training, up the user's choice ?
Author: Loïc Masure
Date: 31/03/2022
"""
from typing import List, Tuple, Dict
import os
import argparse
import logging
from tqdm import tqdm

import datetime

import models
import simulation
from utils import FastTensorDataLoader, epoch_loop
from utils import set_logger, set_deterministic

import numpy as np
import torch

# ##################### Main training loop ####################################
def train_val_loop(model: torch.nn.Module,
                   train_loader: FastTensorDataLoader,
                   test_loader: FastTensorDataLoader,
                   criterion: torch.nn.modules.loss._Loss,
                   num_epochs: int, optimizer, log,
                   white_box: bool = False
                   ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Main loop for training, returns the history of loss w.r.t. both training
    and validation sets.
    """
    history_train: Dict[str, List[float]] = dict()
    history_val_targets: Dict[str, List[float]] = dict()
    history_val_shares: Dict[str, List[float]] = dict()
    best_val = np.inf

    # Main loop
    with tqdm(total=num_epochs) as t:
        for epoch in range(num_epochs):
            train_loss_epoch, _ = epoch_loop(model, train_loader, criterion,
                                             optimizer, train=True, white_box=white_box)

        #     print(train_loss_epoch)
            # Stores the training metrics
            for k, val in train_loss_epoch.items():
                # Handles the potential NaNs in the outputs
                if np.isnan(val).any():
                    log.error("NaN detected in the training loss")
                    break
                if k not in history_train.keys():
                    history_train[k] = [val]
                else:
                    history_train[k].append(val)
            # Storing the training loss for the branches is meaningless

            val_loss_targets, val_loss_shares = epoch_loop(model,
                                               test_loader,
                                               criterion,
                                               optimizer,
                                               train=False)
        #     print(val_loss_targets)
            # Stores the validation metrics
            for k, val in val_loss_targets.items():
                # Handles the potential NaNs in the outputs
                if np.isnan(val).any():
                    log.error("NaN detected in the training loss")
                    break
                if k not in history_val_targets.keys():
                    history_val_targets[k] = [val]
                else:
                    history_val_targets[k].append(val)
            
            for k, val in val_loss_shares.items():
                # Handles the potential NaNs in the outputs
                if np.isnan(val).any():
                    log.error("NaN detected in the training loss")
                    break
                if k not in history_val_shares.keys():
                    history_val_shares[k] = [val]
                else:
                    history_val_shares[k].append(val)

            # Updates the progress bar
            train_loss_mean = np.mean([val for val in train_loss_epoch.values()])
            val_loss_mean = np.mean([val for val in val_loss_targets.values()])
            if val_loss_mean < best_val:
                    best_val = val_loss_mean
            t.set_postfix(train='{:05.3f}'.format(train_loss_mean),
                          val='{:05.3f}'.format(val_loss_mean),
                          best_val='{:05.3f}'.format(best_val))
            t.update()

    return history_train, history_val_targets, history_val_shares
###############################################################################


parser = argparse.ArgumentParser(
        prog="experiment_affine",
        description="""Draws simulated leakages from a model.""",
        epilog="""Author: Loïc Masure (loic.masure@uclouvain.be)""")
parser.add_argument(
        "--n_bits", type=int, default=4,
        help="The number of bits on which the random variables are drawn.")
parser.add_argument(
        "--n_targets", type=int, default=1,
        help="The number of target variables masked with the same masks.")
parser.add_argument(
        "--leakage_model", type=str, default="hw",
        help="The leakage model assumed for the shares.")
parser.add_argument(
        "--sigma", type=float, default=0.1,
        help="The gaussian noise parameter.")
parser.add_argument(
        "--n_draws", type=int, default=1000,
        help="The number of random draws for the simulations.")
parser.add_argument(
        "--n_val", type=int, default=2000,
        help="The number of draws for the validation set.")
parser.add_argument(
        "--seed", type=int, default=123,
        help="To manage the reproducibility of the experiments.")
parser.add_argument(
        "--n_hidden", type=int, default=1000,
        help="The number of neurons in the hidden layer")
parser.add_argument(
        "-normalize_probs", action="store_true",
        help="Whether normalizing the inputs of convolution as probabilities.")
parser.add_argument(
        "--num_epochs", type=int, default=100,
        help="The number of epochs when training the model.")
parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="The learning rate for Adam optimizer.")
parser.add_argument(
        "--batch_size", type=int, default=256,
        help="The batch size for the datasets.")
parser.add_argument(
        "--num_threads", type=int, default=1,
        help="The number of threads when loading the data.")
parser.add_argument(
        "--res_dir", type=str, default="",
        help="The directory to store the results (logs, data).")
parser.add_argument(
        "-debug", action="store_true",
        help="If activated, all the deterministic options are removed.")
args = parser.parse_args()


# Creates an output directory with all the data for post-processing / analysis
if args.res_dir == "":
    date = datetime.datetime.now()
    output_dir = os.path.join(os.getcwd(), "results")
    output_dir = os.path.join(output_dir, str(date))
else:
    output_dir = args.res_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Sets the loggers
log = set_logger("exp.log", output_dir)
log.info("***********SIMULATION Affine Scheme Encoded***********")
log.info("Using Torch: {}".format(torch.__version__))
log.info("Params: {}".format(args.__str__().split('(')[1][:-1]))

# To have a device-agnostic software
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.info("Using the device: {}".format(device))
if torch.cuda.is_available():
    torch.cuda.synchronize()

# Sets the seed for deterministic run
if not args.debug:
    set_deterministic(args.seed, args.num_threads)
    msg = "Deterministic environment set. "
    msg += "Everything should be reproducible now."
    log.info(msg)


# Sets the leakage model for the shares
leak_model = simulation.select_leakage_model(args.leakage_model)
log.info("Selected leakage model: '{}'".format(args.leakage_model))


# Draws the leakage, prepares the data loaders
log.info("Drawing according to the affine scheme")
train_data = simulation.ASCAD(args.n_draws,
                              args.sigma, 
                              args.n_bits,
                              order=2,
                              n_targets=args.n_targets
                              )
val_data = simulation.ASCAD(args.n_val, 
                            args.sigma, 
                            args.n_bits,
                            order=2,
                            n_targets=args.n_targets
                            )
# Computes the entropy
h, h_tot = val_data.entropy()
log.info("Minimum loss/conditional entropy: {}".format(h_tot))

train_loader = FastTensorDataLoader(train_data.leakages,
                                    train_data.labels,
                                    train_data.shares,
                                    batch_size=args.batch_size,
                                    shuffle=True)
test_loader = FastTensorDataLoader(val_data.leakages,
                                   val_data.labels,
                                   val_data.shares,
                                   batch_size=args.n_val,
                                   shuffle=True)
log.info("DataLoaders ready")


# Sets the loss function
criterion = torch.nn.NLLLoss()

###############################################################################
# Creates the model
net = models.BBMLP(
        leakage_dims=train_data.get_shapes(), 
        n_hidden = args.n_hidden * 3, 
        n_bits=args.n_bits).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
log.info("Instanciates a black-box MLP")
log.info(net)

# Training and monitoring
losses_train_Net, losses_val_Net, branch_losses_val_Net = train_val_loop(net,
                                                     train_loader,
                                                     test_loader,
                                                     criterion,
                                                     args.num_epochs,
                                                     optimizer,
                                                     log,
                                                     white_box=False)

# Export history
logging.info("Exporting output data concerning BB-MLP model")
for k, val in losses_train_Net.items():
    filename = os.path.join(output_dir, "losses_train_BB-MLP{}.out".format(k))
    np.savetxt(filename, np.array(val))
for k, val in losses_val_Net.items():
    filename = os.path.join(output_dir, "losses_val_BB-MLP{}.out".format(k))
    np.savetxt(filename, np.array(val))

log.info("End of simulation for BB-MLP model.")
###############################################################################
whnet = models.GroupNet(
        leakage_dims=train_data.get_shapes(), 
        n_hidden = args.n_hidden, 
        scheme="affine", 
        n_bits=args.n_bits, 
        share_weights=False, 
        normalize_probs=args.normalize_probs
        ).to(device)

optimizer = torch.optim.Adam(whnet.parameters(), lr=args.learning_rate)
log.info("Instanciates a WH model")
log.info(whnet)

# Training and monitoring
losses_train, losses_val, branch_losses_val = train_val_loop(whnet,
                                                             train_loader,
                                                             test_loader,
                                                             criterion,
                                                             args.num_epochs,
                                                             optimizer,
                                                             log,
                                                             white_box=False)
# Export history
logging.info("Exporting output data concerning WH model")
for k, val in losses_train.items():
    filename = os.path.join(output_dir, "losses_train_WH_{}.out".format(k))
    np.savetxt(filename, np.array(val))
for k, val in losses_val.items():
    filename = os.path.join(output_dir, "losses_val_WH_{}.out".format(k))
    np.savetxt(filename, np.array(val))
for k, val in branch_losses_val.items():
    filename = os.path.join(output_dir, "branch_losses_val_WH_{}.out".format(k))
    np.savetxt(filename, np.array(val))

log.info("End of simulation for WH model.")
###############################################################################
wbnet = models.GroupNet(
        leakage_dims=train_data.get_shapes(), 
        n_hidden = args.n_hidden, 
        scheme="affine", 
        n_bits=args.n_bits, 
        share_weights=False, 
        normalize_probs=args.normalize_probs
        ).to(device)

optimizer = torch.optim.Adam(wbnet.parameters(), lr=args.learning_rate)
log.info("Instanciates a WB model")
log.info(wbnet)

# Training and monitoring
losses_train, losses_val, branch_losses_val = train_val_loop(wbnet,
                                                             train_loader,
                                                             test_loader,
                                                             criterion,
                                                             args.num_epochs,
                                                             optimizer,
                                                             log,
                                                             white_box=True)
# Export history
logging.info("Exporting output data concerning WB model")
for k, val in losses_train.items():
    filename = os.path.join(output_dir, "losses_train_WB_{}.out".format(k))
    np.savetxt(filename, np.array(val))
for k, val in losses_val.items():
    filename = os.path.join(output_dir, "losses_val_WB_{}.out".format(k))
    np.savetxt(filename, np.array(val))
for k, val in branch_losses_val.items():
    filename = os.path.join(output_dir, "branch_losses_val_WB_{}.out".format(k))
    np.savetxt(filename, np.array(val))

log.info("End of simulation for WB model.")
###############################################################################

log.info("End of simulation")

