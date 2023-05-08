@experiment(backend="csv", save_to=results_dir, verbose=True)
def train_model(sigma, cutoff, noise):

    n_max = 16
    l_max = 8

    print(f"n_max = {n_max}, l_max = {l_max}")

    desc = Descriptor(
        f"soap n_max={n_max} l_max={l_max} cutoff={cutoff} atom_sigma={sigma} average=F n_Z=1 Z={{30}} n_species=4 species_Z={{30 7 6 1}}"
    )

    av_train_rmse, av_test_rmse, train_predictions, test_predictions = gpr_with_cv(
        noise,
        complete_cg_df,
        fold_ids,
        train_batches=all_rattled_batches,
        test_batches=all_rattled_batches,
        descriptor=desc,
        atomistic_df=complete_a_df,
        numb_train=12000,
    )

    return {"av_train_rmse": av_train_rmse, "av_test_rmse": av_test_rmse}
