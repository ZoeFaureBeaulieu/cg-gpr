@experiment(
    backend="csv", save_to=f"grid_search_{struct_type}_{numb_train}", verbose=True
)
def train_model(sigma, cutoff, noise=noise):
    n_max = 16
    l_max = 8  # based on learning curves, no need to go higher

    print(f"{struct_type} Grid Search")
    print(f"n_max = {n_max}, l_max = {l_max}")
    # print(f"noise: {noise:.2f}, sigma: {sigma}, cutoff: {cutoff}")

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
        a_df = complete_a_df

    av_train_rmse, av_test_rmse, train_predictions, test_predictions = gpr_with_cv(
        noise,
        complete_cg_df,
        fold_ids,
        train_batches=all_rattled_batches,
        test_batches=all_rattled_batches,
        descriptor=desc,
        atomistic_df=a_df,
        numb_train=numb_train,
        B_site=B_site,
    )

    return {"av_train_rmse": av_train_rmse, "av_test_rmse": av_test_rmse}
