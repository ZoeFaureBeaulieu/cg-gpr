from pathlib import Path
from data import (
    get_complete_dataframes,
    get_fold_ids,
    get_opt_hypers,
    get_opt_soap_descriptor,
    get_energies,
    calc_soap_vectors,
)
from gpr_functions import train_gpr, predict_gpr
import argparse
import numpy as np
import random
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent


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
    "--l_max", type=int, help="l_max for SOAP", default=8
)  # l_max is optional; default is 8 based on convergence tests, no need to go higher. A lower l_max will be faster and less memory intensive.
parser.add_argument(
    "--energy_type", type=str, help="Energy type", default="e_local_mofff"
)
args = parser.parse_args()

print(f"Structure type: {args.struct_type}")

all_rattled_batches = [2, 3, 4, 5]
energy_cutoff = 1
numb_train = 10

print(f"Number of training atoms: {numb_train}")

complete_cg_df, complete_a_df = get_complete_dataframes(energy_cutoff=1)

train_cg_df = complete_cg_df.xs("medium", level=1)
print(f"training dataframe shape: {train_cg_df.shape}")
test_cg_df1 = complete_cg_df.xs("small", level=1)
test_cg_df2 = complete_cg_df.xs("large", level=1)

# combine the two test sets
test_cg_df = pd.concat([test_cg_df1, test_cg_df2])
print(f"test dataframe shape: {test_cg_df.shape}")

soap_cutoff, atom_sigma, noise = get_opt_hypers(args.struct_type)
desc, noise = get_opt_soap_descriptor(args.struct_type)
print(f"SOAP cutoff: {soap_cutoff}; atom_sigma: {atom_sigma}; noise: {noise}")
# set the B_site flag and the atomistic dataframe if needed
if args.struct_type == "cg":
    B_site = True
    train_a_df = None
    test_a_df = None
elif args.struct_type == "A_cg":
    B_site = False
    train_a_df = None
    test_a_df = None
else:
    B_site = True
    train_a_df = complete_a_df.xs("medium", level=1)
    test_a_df1 = complete_a_df.xs("small", level=1)
    test_a_df2 = complete_a_df.xs("large", level=1)
    test_a_df = pd.concat([test_a_df1, test_a_df2])
    print(f"training atomistic dataframe shape: {train_a_df.shape}")
    print(f"test atomistic dataframe shape: {test_a_df.shape}")

struct_ids = get_fold_ids(train_cg_df, 1)

train_energies = get_energies(
    train_cg_df, struct_ids, all_rattled_batches, energy_type=args.energy_type
)
test_energies = get_energies(
    test_cg_df, struct_ids, all_rattled_batches, energy_type=args.energy_type
)

# if atomistic_df is not None, use the atomistic SOAPs for training
if train_a_df is not None:
    train_soaps = calc_soap_vectors(
        train_a_df, struct_ids, desc, all_rattled_batches, B_site=B_site
    )
    test_soaps = calc_soap_vectors(
        test_a_df, struct_ids, desc, all_rattled_batches, B_site=B_site
    )
# else use the coarse-grained SOAPs
else:
    train_soaps = calc_soap_vectors(
        train_cg_df, struct_ids, desc, all_rattled_batches, B_site=B_site
    )
    test_soaps = calc_soap_vectors(
        test_cg_df, struct_ids, desc, all_rattled_batches, B_site=B_site
    )

train_energies = np.concatenate(train_energies).reshape(-1, 1)
test_energies = np.concatenate(test_energies).reshape(-1, 1)
print(f"Number of test atoms: {len(test_energies)}")

train_soaps = np.concatenate(train_soaps)
test_soaps = np.concatenate(test_soaps)


assert len(train_soaps) == len(train_energies)
assert len(test_soaps) == len(test_energies)

if numb_train is not None:
    random.seed(42)
    list = random.sample(range(0, len(train_energies)), numb_train)
    train_energies = train_energies[list]
    train_soaps = train_soaps[list]

model = train_gpr(noise, train_soaps, train_energies)

train_predictions, train_error = predict_gpr(model, train_soaps, train_energies)
test_predictions, test_error = predict_gpr(model, test_soaps, test_energies)

results = {
    "train_predictions": train_predictions,
    "train_energies": train_energies,
    "test_predictions": test_predictions,
    "test_energies": test_energies,
}

print(f"Training error: {train_error:.3f}")
print(f"Test error: {test_error:.3f}")

np.save(
    root_dir
    / f"results/external_test/medium_train_{args.struct_type}_preds_ntrain{numb_train}.npy",
    results,
)
