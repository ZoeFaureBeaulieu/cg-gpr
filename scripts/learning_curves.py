from pathlib import Path
from data import (
    get_complete_dataframes,
    get_fold_ids,
    get_opt_hypers,
    get_opt_soap_descriptor,
)
from gpr_functions import gpr_with_cv
import csv
import argparse

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
    "--linker_type",
    type=str,
    help="Linker type (H or CH3)",
    required=True,
)

# optional arguments
parser.add_argument(
    "--l_max", type=int, help="l_max for SOAP", default=8
)  # l_max is optional; default is 8 based on convergence tests, no need to go higher. A lower l_max will be faster and less memory intensive.
parser.add_argument(
    "--energy_type", type=str, help="Energy type", default="e_local_mofff"
)
args = parser.parse_args()

print(f"Structure type: {args.struct_type}")

if args.linker_type == "H":
    all_rattled_batches = [2, 3, 4, 5, 6]
    energy_cutoff = 1
elif args.linker_type == "CH3":
    all_rattled_batches = [2, 3, 4, 5]
    energy_cutoff = -5.7
elif args.linker_type == "H_new":
    all_rattled_batches = [2, 3, 4, 5]
    energy_cutoff = 1

# load all the data as two dataframes: one for the cg structures and one for the atomistic structures
complete_cg_df, complete_a_df = get_complete_dataframes(
    energy_cutoff=energy_cutoff, im_linker=args.linker_type
)

# randomly split the structure ids into k folds
fold_ids = get_fold_ids(complete_cg_df, 5)

# number of training environments to use
n_train = [7, 15, 31, 62, 125, 250, 500, 1000, 2000, 4000, 8000]

# get the SOAP descriptor with optimised hyperparameters
# and the optimised regularisation noise
soap_cutoff, atom_sigma, noise = get_opt_hypers(args.struct_type, args.linker_type)
desc, _ = get_opt_soap_descriptor(args.struct_type, args.l_max)

print(f"Cutoff={soap_cutoff}; sigma={atom_sigma}; noise={noise}")

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

# create a csv file to store the results
file_name = (
    root_dir
    / f"results/new_learning_curve/lc_lMax{args.l_max}_{args.struct_type}_{args.linker_type}.csv"
)

headers = [
    "soap_cutoff",
    "atom_sigma",
    "noise",
    "numb_training_atoms",
    "av_train_rmse",
    "av_test_rmse",
]

if not Path(file_name).exists():
    with open(file_name, "a") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

# train and evaluate the model
with open(file_name, "a") as f:
    writer = csv.writer(f)
    # loop over the number of training environments
    for i in n_train:
        print(f"Number of training environments: {i}")
        av_train_rmse, av_test_rmse, _, _, _ = gpr_with_cv(
            noise,
            complete_cg_df,
            fold_ids,
            train_batches=all_rattled_batches,
            test_batches=all_rattled_batches,
            descriptor=desc,
            numb_train=i,
            atomistic_df=a_df,
            B_site=B_site,
            energy_type=args.energy_type,
        )
        print(f"Average training RMSE: {av_train_rmse}")
        print(f"Average test RMSE: {av_test_rmse}")
        row = [soap_cutoff, atom_sigma, noise, i, av_train_rmse, av_test_rmse]
        writer.writerow(row)
