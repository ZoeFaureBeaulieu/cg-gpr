from digital_experiments import experiment
from digital_experiments.optmization import optimize_step_for, Real
from data import (
    get_complete_dataframes,
    build_soap_descriptor,
    get_fold_ids,
    get_opt_hypers,
)
from gpr_functions import gpr_with_cv
import argparse
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent

# get hyperparameters from the command line
parser = argparse.ArgumentParser()
parser.add_argument(
    "--struct_type",
    type=str,
    help="Structure type (cg, A_cg or atomistic)",
    required=True,
)
parser.add_argument(
    "--medium_only",
    type=bool,
    help="Use medium-rattled structures only",
    required=True,
)

# optional arguments
parser.add_argument(
    "--numb_train", type=int, help="Number of training atoms", default=1000
)  # number of training environments is optional; default is 20000
parser.add_argument(
    "--energy_type", type=str, help="Energy type", default="e_local_mofff"
)
args = parser.parse_args()

all_rattled_batches = [2, 3, 4, 5]
energy_cutoff = 1

# load all the data as two dataframes: one for the cg structures and one for the atomistic structures
complete_cg_df, complete_a_df = get_complete_dataframes(energy_cutoff=energy_cutoff)
if args.medium_only:
    complete_cg_df = complete_cg_df.xs("medium", level=1)
    complete_a_df = complete_a_df.xs("medium", level=1)

print(f"Dataframe shape: {complete_cg_df.shape}")

# randomly split the structure ids into k folds
fold_ids = get_fold_ids(complete_cg_df, 5)

# use digital experiments package to save the results to a csv file
if args.medium_only:
    results_dir = root_dir / f"results/medium_only_hypop/{args.struct_type}"
else:
    results_dir = root_dir / f"results/hypop/{args.struct_type}"

soap_cutoff, atom_sigma, _ = get_opt_hypers(args.struct_type)


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
    # "cutoff": Real(1.5, 15),
    # "sigma": Real(0.1, 2),
    "noise": Real(1e-4, 1, prior="log-uniform"),  # use a log scale for the noise
}

# run the optimisation using the digital experiments package
for _ in range(100):
    optimize_step_for(
        train_model,
        config_overides={"atom_sigma": atom_sigma, "soap_cutoff": soap_cutoff},
        # minimize the average test set RMSE
        loss_fn=lambda results: results["av_test_rmse"],
        n_random_points=70,  # number of random points to try before Bayesian optimisation
        space=search_space,
        root=results_dir,
    )
