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
