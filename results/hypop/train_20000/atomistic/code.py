@experiment(backend="csv", save_to=results_dir, verbose=True)
def train_model(sigma, cutoff, noise):

    n_max = 16
    l_max = 8

    print(f"n_max = {n_max}, l_max = {l_max}")

    if struct_type == "cg":
        desc = Descriptor(
            f"soap n_max={n_max} l_max={l_max} cutoff={cutoff} atom_sigma={sigma} average=F n_Z=1 Z={{14}} n_species=2 species_Z={{14 8}}"
        )
        B_site = True
        a_df = None

    elif struct_type == "A_cg":
        desc = Descriptor(
            f"soap n_max={n_max} l_max={l_max} cutoff={cutoff} atom_sigma={sigma} average=F n_Z=1 Z={{14}}"
        )
        B_site = False
        a_df = None

    else:
        desc = Descriptor(
            f"soap n_max={n_max} l_max={l_max} cutoff={cutoff} atom_sigma={sigma} average=F n_Z=1 Z={{30}} n_species=4 species_Z={{30 7 6 1}}"
        )
        B_site = True
        a_df = concat_dataframes([a_mofs, a_zeolites])

    av_train_rmse, av_test_rmse, train_predictions, test_predictions = gpr_with_cv(
        noise,
        cg_df,
        fold_ids,
        train_batches=all_rattled_batches,
        test_batches=all_rattled_batches,
        descriptor=desc,
        numb_train=numb_train,
        atomistic_df=a_df,
        B_site=B_site,
    )

    return {"av_train_rmse": av_train_rmse, "av_test_rmse": av_test_rmse}
