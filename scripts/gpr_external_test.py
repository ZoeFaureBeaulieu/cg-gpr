from pathlib import Path
from data import (
    get_complete_dataframes,
    get_fold_ids,
    get_opt_hypers,
    get_opt_soap_descriptor,
    get_energies,
    calc_soap_vectors,
    remove_B_sites,
)
from gpr_functions import train_gpr, predict_gpr
import argparse
from ase.io import read
import numpy as np
import random

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

numb_train = 32000

complete_cg_df, complete_a_df = get_complete_dataframes(
    energy_cutoff=energy_cutoff, im_linker=args.linker_type
)

print(f"Number of training atoms: {numb_train}")

test_atomistic = read(root_dir / "bulk_modulus_data/atomistic.xyz", index=":")
test_cg = read(root_dir / "bulk_modulus_data/coarse_grained.xyz", index=":")

soap_cutoff, atom_sigma, noise = get_opt_hypers(args.struct_type, linker_type="H")
desc, noise = get_opt_soap_descriptor(args.struct_type, l_max=8)

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

struct_ids = get_fold_ids(complete_cg_df, 1)

train_energies = get_energies(
    complete_cg_df, struct_ids, all_rattled_batches, energy_type=args.energy_type
)

# if atomistic_df is not None, use the atomistic SOAPs for training
if a_df is not None:
    train_soaps = calc_soap_vectors(
        a_df, struct_ids, desc, all_rattled_batches, B_site=B_site
    )
# else use the coarse-grained SOAPs
else:
    train_soaps = calc_soap_vectors(
        complete_cg_df, struct_ids, desc, all_rattled_batches, B_site=B_site
    )

train_energies = np.concatenate(train_energies).reshape(-1, 1)
train_soaps = np.concatenate(train_soaps)
assert len(train_soaps) == len(train_energies)

if numb_train is not None:
    random.seed(42)
    list = random.sample(range(0, len(train_energies)), numb_train)
    train_energies = train_energies[list]
    train_soaps = train_soaps[list]

model = train_gpr(noise, train_soaps, train_energies)

train_predictions, train_error = predict_gpr(model, train_soaps, train_energies)

print(f"Training error: {train_error:.3f}")

per_struct_preds = []
per_struct_rmses = []

test_energies = []
for s in test_cg:
    test_energies.append(s.arrays[f"{args.energy_type}"][s.numbers == 14])

if args.struct_type == "cg":
    test_structures = test_cg
elif args.struct_type == "atomistic":
    test_structures = test_atomistic
elif args.struct_type == "A_cg":
    test_structures = []
    for s in test_cg:
        new_s = remove_B_sites(s)
        test_structures.append(new_s)

for s in range(len(test_structures)):
    test_e = test_energies[s]
    test_vectors = desc.calc(test_structures[s])["data"]

    assert len(test_e) == len(test_vectors)

    test_predictions, test_error = predict_gpr(model, test_vectors, test_e)
    per_struct_preds.append(test_predictions)
    per_struct_rmses.append(test_error)

np.save(
    root_dir
    / f"results/bulk_modulus/{args.struct_type}_BM_preds_ntrain{numb_train}_H_new.npy",
    per_struct_preds,
)
