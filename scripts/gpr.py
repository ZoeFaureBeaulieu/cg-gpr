import numpy as np
from data import (
    get_complete_dataframes,
    get_fold_ids,
    get_opt_hypers,
    build_soap_descriptor,
)
from gpr_functions import gpr_with_cv
from pathlib import Path
import argparse

root_dir = Path(__file__).resolve().parent.parent
gpr_results = root_dir / "results/new_gpr"

# get hyperparameters from the command line
parser = argparse.ArgumentParser()
parser.add_argument(
    "--struct_type",
    type=str,
    help="Structure type (cg, A_cg or atomistic)",
    required=True,
)
parser.add_argument(
    "--hypers_type",
    type=str,
    help="The structure type from which to take the optimised hypers",
    required=True,
)


# optional arguments
parser.add_argument(
    "--numb_train", type=int, help="Number of training atoms", default=32000
)  # number of training environments; default is 32000
parser.add_argument(
    "--l_max", type=int, help="l_max for SOAP", default=8
)  # l_max; default is 8 based on convergence tests, no need to go higher. A lower l_max will be faster and less memory intensive.
parser.add_argument(
    "--energy_type", type=str, help="Energy type", default="e_local_mofff"
)
args = parser.parse_args()

print(f"Structure type: {args.struct_type}")
print(f"Hypers type: {args.hypers_type}")
print(f"Number of training atoms: {args.numb_train}")

all_rattled_batches = [2, 3, 4, 5]
energy_cutoff = 1

# load all the data as two dataframes: one for the cg structures and one for the atomistic structures
complete_cg_df, complete_a_df = get_complete_dataframes(energy_cutoff=energy_cutoff)

# randomly split the structure ids into k folds
fold_ids = get_fold_ids(complete_cg_df, 5)

# get the SOAP descriptor with optimised hyperparameters
# and the optimised regularisation noise
soap_cutoff, atom_sigma, noise = get_opt_hypers(args.hypers_type)
desc = build_soap_descriptor(args.struct_type, soap_cutoff, atom_sigma, args.l_max)


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
(
    av_train_rmse,
    av_test_rmse,
    train_predictions,
    test_predictions,
    test_labels,
) = gpr_with_cv(
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
print(f"Average training RMSE: {av_train_rmse}")
print(f"Average test RMSE: {av_test_rmse}")

test_preds = np.concatenate(test_predictions)
train_preds = np.concatenate(train_predictions)
test_labels = np.concatenate(test_labels)

# save the results to a numpy file
gpr_preds = {
    "av_test_rmse": av_test_rmse,
    "test_predictions": test_preds,
    "av_train_rmse": av_train_rmse,
    "train_predictions": train_preds,
    "test_labels": test_labels,
}

if args.energy_type == "e_local_mofff":
    e_label = "local_energies"
else:
    e_label = "zn_energies"

np.save(
    gpr_results
    / e_label
    / f"{args.hypers_type}_hypers/gpr_{args.struct_type}_ntrain{args.numb_train}.npy",
    gpr_preds,
)
