@experiment(backend="csv", save_to=results_dir, verbose=True)
def train_model(sigma, cutoff, noise):

    n_max = 16
    l_max = 8

    print(f"n_max = {n_max}, l_max = {l_max}")

    desc = Descriptor(
        f"soap n_max={n_max} l_max={l_max} cutoff={cutoff} atom_sigma={sigma} average=F n_Z=1 Z={{14}} n_species=2 species_Z={{14 8}}"
    )

    av_train_rmse, av_test_rmse, train_predictions, test_predictions = gpr_with_cv(
        noise,
        cg_df,
        fold_ids,
        train_batches=all_rattled_batches,
        test_batches=all_rattled_batches,
        descriptor=desc,
        numb_train=20000,
    )

    return {"av_train_rmse": av_train_rmse, "av_test_rmse": av_test_rmse}
