from digital_experiments import experiment
from digital_experiments import optimize_step_for, Real
from data import (
    get_complete_dataframes,
    build_soap_descriptor,
    get_fold_ids,
    all_rattled_batches,
)
from gpr_functions import gpr_with_cv
import argparse

# get hyperparameters from the command line
parser = argparse.ArgumentParser()
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

# use digital experiments package to save the results to a csv file
results_dir = f"results/hypop/train_{args.numb_train}/{args.struct_type}"


@experiment(backend="csv", save_to=results_dir, verbose=True)
def train_model(atom_sigma: float, soap_cutoff: float, noise: float) -> dict:
    """Train a GPR model with the given SOAP hyperparameters and evaluate it using cross-validation.

    Args:
        atom_sigma (float): sigma parameter for the SOAP descriptor.
        soap_cutoff (float): cutoff used to build the SOAP descriptor.
        noise (float): The per-atom regularisaton term.

    Returns:
        dict: the average training and test RMSEs, averaged over the k folds
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

    # train and evaluate the model
    av_train_rmse, av_test_rmse, _, _, _ = gpr_with_cv(
        noise,
        complete_cg_df,
        fold_ids,
        train_batches=all_rattled_batches,
        test_batches=all_rattled_batches,
        descriptor=desc,
        numb_train=args.numb_train,
        atomistic_df=a_df,
        B_site=B_site,
        energy_type=args.energy_type,
    )

    return {"av_train_rmse": av_train_rmse, "av_test_rmse": av_test_rmse}


# define the search space for the optimisation
search_space = {
    "cutoff": Real(1.5, 15),
    "sigma": Real(0.1, 2),
    "noise": Real(1e-4, 1, prior="log-uniform"),  # use a log scale for the noise
}

# run the optimisation using the digital experiments package
optimize_step_for(
    train_model,
    # minimize the average test set RMSE
    loss_fn=lambda results: results["av_test_rmse"],
    n_random_points=70,  # number of random points to try before Bayesian optimisation
    space=search_space,
    root=results_dir,
)
