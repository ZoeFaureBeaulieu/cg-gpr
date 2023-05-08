from data import (
    get_complete_dataframes,
    get_fold_ids,
    all_rattled_batches,
    get_opt_hypers,
    build_soap_descriptor,
)
from gpr_functions import gpr_with_cv
import argparse
from itertools import product
from digital_experiments import experiment
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent

# get hyperparameters from the command line
# index controls which point on the grid to run
parser = argparse.ArgumentParser()
parser.add_argument(
    "--index", type=int, help="Index of the grid point to run", required=True
)
parser.add_argument(
    "--struct_type",
    type=str,
    help="Structure type (cg, A_cg or atomistic)",
    required=True,
)

# optional arguments
parser.add_argument(
    "--numb_train", type=int, help="Number of training atoms", default=20000
)  # number of training environments is optional; default is 20000
parser.add_argument(
    "--energy_type", type=str, help="Energy type", default="e_local_mofff"
)
args = parser.parse_args()

# load all the data as two dataframes: one for the cg structures and one for the atomistic structures
complete_cg_df, complete_a_df = get_complete_dataframes(energy_cutoff=1)

# randomly split the structure ids into k folds
fold_ids = get_fold_ids(complete_cg_df, 5)

# get the optimised regularisation noise, obtained from Bayesian optimisation
_, _, noise = get_opt_hypers(args.struct_type)

# using the digital experiments package to save the results to a csv file
results_path = root_dir / f"results/grid_search/{args.struct_type}"


@experiment(backend="csv", save_to=results_path, verbose=True)
def train_model(atom_sigma: float, soap_cutoff: float, noise: float = noise) -> dict:
    """Train a GPR model with the given SOAP hyperparameters and evaluate it using cross-validation.

    Args:
        atom_sigma (float): sigma parameter for the SOAP descriptor.
        soap_cutoff (float): cutoff used to build the SOAP descriptor.
        noise (float, optional): The per-atom regularisaton term. Defaults to the optimal noise obtained from Bayesian optimisation.

    Returns:
        dict: the average training and test RMSEs, averaged over the k folds.
    """
    l_max = 8  # Based on convergence tests, no need to go higher. A lower l_max will be faster and less memory intensive.

    desc = build_soap_descriptor(args.struct_type, soap_cutoff, atom_sigma, l_max)

    # set the B_site flag and the atomistic dataframe if needed
    if args.struct_type == "cg":
        B_site = True
        a_df = None
    elif args.struct_type == "A_cg":
        B_site = False
        a_df = None
    else:
        B_site = True
        a_df = complete_a_df

    # train and evaluate the model using cross-validation
    av_train_rmse, av_test_rmse, _, _, _ = gpr_with_cv(
        noise,
        complete_cg_df,
        fold_ids,
        train_batches=all_rattled_batches,
        test_batches=all_rattled_batches,
        descriptor=desc,
        atomistic_df=a_df,
        numb_train=args.numb_train,
        B_site=B_site,
        energy_type=args.energy_type,
    )

    return {"av_train_rmse": av_train_rmse, "av_test_rmse": av_test_rmse}


# define the grid of hyperparameters to search over
sigmas = [
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.25,
    1.5,
    1.75,
    2,
]
cutoffs = [
    2.5,
    3,
    3.5,
    4,
    4.5,
    5,
    5.25,
    5.5,
    5.75,
    6,
    6.25,
    6.5,
    6.75,
    7,
    7.25,
    7.5,
    7.75,
    8,
    8.5,
    9,
    9.5,
    10,
    11,
    12,
    13,
    14,
    15,
]

# create the grid of hyperparameters
# total number of jobs = len(cutoffs) * len(sigmas)
grid = list(product(sigmas, cutoffs))

# get the hyperparameters for the current job; index controls which point on the grid to run and is set in the command line
atom_sigma, soap_cutoff = grid[args.index - 1]

# train the model and save the results
av_train_rmse, av_test_rmse = train_model(atom_sigma, soap_cutoff)
