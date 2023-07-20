from ase.io import read
import numpy as np
import math
import random
from pathlib import Path
import pandas as pd
from typing import List, Tuple
from quippy.descriptors import Descriptor
from ase import Atoms

# dataset names
MOF = "AB2_MOF"
ZEOLITE = "Zeolites"

# paths to the dataset and grid search results directories
root_dir = Path(__file__).resolve().parent.parent
hZIF_data = root_dir / "hZIF-data"
grid_search_results = root_dir / "results/grid_search"
new_grid_search_results = root_dir / "results/new_grid_search"

# dict of the rattling parameters used
r_levels = {
    1: {"rms": 0.001, "length": 0.005, "angle": 0.25, "rattling_type": "small"},
    2: {"rms": 0.01, "length": 0.05, "angle": 2.5, "rattling_type": "medium"},
    3: {"rms": 0.1, "length": 0.1, "angle": 5.0, "rattling_type": "large"},
}

# the two ways of generating the rattled structures: relax then decorate or decorate then relax
gen_codes = ["r-d", "d-r"]

# batch numbers associated with each set of structures
# batch 1 is the original dataset, i.e. no rattling
# batches 2-6 are the rattled datasets
# each batch contains contains 6x the number of structures as batch 1, for each pair of gen_codes and r_levels, i.e. r_level[1]x"r-d", r_levels[1]x"d-r" and so on
# each batch 2-6 contains the same set of structures but each rattled with a different random seed
# the random seed is used to select which atoms are rattled
H_rattled_batches = [2, 3, 4, 5, 6]
CH3_rattled_batches = [2, 3, 4, 5]


def get_complete_dataframes(
    energy_cutoff: int, im_linker: str = "H", curated: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get the complete coarse-grained and atomistic dataframes containing both the MOF and Zeolite datasets.
    These dataframes have been processed to remove high energy structures and null columns.

    Args:
        energy_cutoff (int): the energy cutoff used to remove high energy structures. Any structures containing Si atoms with local energy above this cutoff are removed.
        im_linker (str, optional): the imidazolate linker used to generate the structures. Either "H" or "CH3".
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: the complete processed dataframes with either the coarse-grained or atomistic atoms objects.
    """
    # obtain and process the MOF structures
    cg_mofs = get_all_data(MOF, coarse_grain=True, im_linker=im_linker)
    if im_linker == "H":
        a_mofs = get_all_data(MOF, coarse_grain=False, im_linker=im_linker)
    else:
        a_mofs = get_curated_data(MOF, coarse_grain=False, im_linker=im_linker)
    if curated:
        remove_close_contacts(cg_mofs, a_mofs)
    # else:
    #     a_mofs = get_all_data(MOF, coarse_grain=False, im_linker=im_linker)

    remove_high_energy_structures(
        cg_mofs, energy_cutoff=energy_cutoff, atomistic_df=a_mofs
    )

    # cg_mofs, a_mofs = remove_null_columns(cg_mofs, a_mofs)

    # obtain and process the Zeolite structures
    cg_zeolites = get_all_data(ZEOLITE, coarse_grain=True, im_linker=im_linker)
    if im_linker == "H":
        a_zeolites = get_all_data(ZEOLITE, coarse_grain=False, im_linker=im_linker)
    else:
        a_zeolites = get_curated_data(ZEOLITE, coarse_grain=False, im_linker=im_linker)
    if curated:
        remove_close_contacts(cg_zeolites, a_zeolites)

    remove_high_energy_structures(
        cg_zeolites, energy_cutoff=energy_cutoff, atomistic_df=a_zeolites
    )

    # cg_zeolites, a_zeolites = remove_null_columns(cg_zeolites, a_zeolites)

    # concatenate the MOF and Zeolite dataframes
    complete_cg_df = concat_dataframes([cg_mofs, cg_zeolites])
    complete_a_df = concat_dataframes([a_mofs, a_zeolites])

    return complete_cg_df, complete_a_df


def get_file_identifier(
    batch_number: int,
    coarse_grain: bool,
    im_linker: str = "H",
    rms: float = None,
    length: float = None,
    angle: float = None,
    rattling_type: str = None,
    gen_code: str = None,
) -> str:
    """Get the file identifier for a given set of rattling parameters.

    Args:
        batch_number (int): the batch number we want the structures from
        coarse_grain (bool): coarse-grained or atomistic structures
        im_linker (str, optional): the linker type. Defaults to "H". Options are 'H' or 'CH3'.
        rms (float, optional): rms displacement on rattling. Defaults to None. Only used for batches 2-6.
        length (float, optional): %change in cell length. Defaults to None. Only used for batches 2-6.
        angle (float, optional): perturbation to cell angles in degrees. Defaults to None. Only used for batches 2-6.
        rattling_type (str, optional): the rattling level. Defaults to None. Options are 'small', 'medium' or 'large'.
        gen_code (str, optional): the way in which the rattled structures were generated. Defaults to None. Only used for batches 2-6.

    Returns:
        str: the file extension
    """
    if im_linker == "H":
        if coarse_grain:
            if batch_number == 1:
                return f"coarse-grained/batch1_d-rlx-cg-d-cg"
            else:
                return f"coarse-grained/batch{batch_number}_d-rlx-cg-{gen_code}_rms{rms}_length{length}_angle{angle}-cg"
        else:
            if batch_number == 1:
                return f"atomistic/batch1_d-rlx-cg-d"
            else:
                return f"atomistic/batch{batch_number}_d-rlx-cg-{gen_code}_rms{rms}_length{length}_angle{angle}"

    else:
        if coarse_grain:
            if batch_number == 1:
                return f"coarse-grained/batch1_d-rlx-cg-d"
            else:
                return f"coarse-grained/batch{batch_number}_d-rlx-cg-{gen_code}_{rattling_type}"
        else:
            if batch_number == 1:
                return f"atomistic/batch1_d-rlx-cg-d"
            else:
                return (
                    f"atomistic/batch{batch_number}_d-rlx-cg-{gen_code}_{rattling_type}"
                )


def get_all_data(
    struct_type: str, coarse_grain: bool, im_linker: str = "H"
) -> pd.DataFrame:
    """Organises all the structures in the database into a dataframe.
    Rows are indexed according to the following parameters:
    - batch: the batch number, i.e 1, 2, 3, 4, 5 or 6
    - rattling: the rattling level, i.e. 1, 2 or 3
    - gen-code: the way in which the rattled structures were generated i.e. relax then decorate (r-d) or decorate then relax (d-r)
    Each column corresponds to a given structure ID tag.
    Each cell contains a structure as an ase.Atoms object.
    Each atoms object can be identified by its unique combination of batch, rattling, gen-code and id.

                                 |               id tag
    +------+----------+----------+------------+-----------+-----------+
    |batch | rattling | gen-code |  AB1_MOF-1 | AB1_MOF-2 | AB1_MOF-3 |
    +------+----------+----------+------------+-----------+-----------+
    |  1   |   None   | None     |  Atoms     | Atoms     | Atoms     |
    +------+----------+----------+------------+-----------+-----------+
    |      |          | d-r      |  Atoms     | Atoms     | Atoms     |
    |      |   small  +----------+------------+-----------+-----------+
    |      |          | r-d      |  Atoms     | Atoms     | Atoms     |
    |      +----------+----------+------------+-----------+-----------+
    |      |          | d-r      |  Atoms     | Atoms     | Atoms     |
    |  2   |  medium  +----------+------------+-----------+-----------+
    |      |          | r-d      |  Atoms     | Atoms     | Atoms     |
    |      +----------+----------+------------+-----------+-----------+
    |      |          | d-r      |  Atoms     | Atoms     | Atoms     |
    |      |   large  +----------+------------+-----------+-----------+
    |      |          | r-d      |  Atoms     | Atoms     | Atoms     |
    +------+----------+----------+------------+-----------+-----------+

    Args:
        struct_type (str): MOF or Zeolite
        coarse_grain (bool): coarse-grained or atomistic structures
        im_linker (str, optional): the linker type. Defaults to "H". Options are 'H' or 'CH3'.

    Returns:
        pd.Dataframe: all structures organised into a dataframe.
    """
    s_list = []

    if im_linker == "H":
        all_batches = [1, 2, 3, 4, 5, 6]
    else:
        all_batches = [1, 2, 3, 4, 5]

    # loop through all batches
    for b in all_batches:
        # batch 1 is not rattled
        if b == 1:
            file_id = get_file_identifier(
                coarse_grain=coarse_grain, batch_number=b, im_linker=im_linker
            )
            structures = read(
                hZIF_data / f"{struct_type}/MOFFF/{im_linker}/{file_id}.xyz",
                index=":",
            )
            # loop through all structures in batch 1 (i.e. no rattling)
            # label each structure with its batch number, rattling level, gen-code and ID tag
            for s in structures:
                s_list.append(
                    {
                        "batch": b,
                        "rattling": 0,
                        "gen-code": "None",
                        "id": s.info["id"],
                        "structure": s,  # the structure as an ase.Atoms object
                    }
                )
        # batches 2-6 are rattled
        else:
            # loop through the 3 rattling levels
            for r in r_levels:
                r_params = r_levels[r]
                # loop through the 2 gen-codes
                for g in gen_codes:
                    file_id = get_file_identifier(
                        **r_params,
                        gen_code=g,
                        coarse_grain=coarse_grain,
                        batch_number=b,
                        im_linker=im_linker,
                    )
                    structures = read(
                        hZIF_data / f"{struct_type}/MOFFF/{im_linker}/{file_id}.xyz",
                        index=":",
                    )
                    # loop through all structures in the batch
                    # label each structure with its batch number, rattling level, gen-code and ID tag
                    for s in structures:
                        s_list.append(
                            {
                                "batch": b,
                                "rattling": r,
                                "gen-code": g,
                                "id": s.info["id"],
                                "structure": s,
                            }
                        )

    # convert the list of dictionaries into a dataframe
    df = pd.DataFrame(s_list)

    # pivot the dataframe so that each column corresponds to a given structure ID tag
    df = df.pivot(
        index=["batch", "rattling", "gen-code"],
        columns="id",
        values="structure",
    )
    return df


def remove_high_energy_structures(
    cg_df: pd.DataFrame, energy_cutoff: int, atomistic_df: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Removes structures which contain any Si atoms with local energies above a given cutoff.

    Args:
        cg_df (pd.DataFrame): complete dataframe of coarse-grained structures
        energy_cutoff (int): local energy cutoff
        atomistic_df (pd.DataFrame, optional): dataframe of atomistic structures. Defaults to None. If given, the atomistic dataframe will also be modified to remove the corresponding structures.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: the coarse-grained and atomistic dataframes with the high energy structures removed and replaced with NaNs
    """
    # if energy_cutoff is None, set it to infinity
    if energy_cutoff is None:
        energy_cutoff = math.inf

    for rowIndex, row in cg_df.iterrows():
        for (
            columnIndex,
            s,
        ) in row.items():
            # if the cell is empty, skip it
            if cg_df.loc[rowIndex, columnIndex] is np.nan:
                continue
            else:
                # if any Si atoms have local energies above the cutoff, replace the structure with NaN
                if any(s.arrays["e_local_mofff"][s.numbers == 14] > energy_cutoff):
                    cg_df.loc[rowIndex, columnIndex] = np.nan
                    # if an atomistic dataframe is given, replace the corresponding structure with NaN
                    if atomistic_df is not None:
                        atomistic_df.loc[rowIndex, columnIndex] = np.nan
    # return cg_df, atomistic_df


def remove_close_contacts(
    cg_df: pd.DataFrame,
    atomistic_df: pd.DataFrame,
):
    for rowIndex, row in atomistic_df.iterrows():
        for columnIndex, s in row.items():
            # if the cell is empty, skip it
            if atomistic_df.loc[rowIndex, columnIndex] is np.nan:
                continue
            else:
                if s.info["reject_bond"] == True:
                    atomistic_df.loc[rowIndex, columnIndex] = np.nan
                    cg_df.loc[rowIndex, columnIndex] = np.nan
        # return cg_df, atomistic_df


def remove_null_columns(
    cg_df: pd.DataFrame, a_df: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove any columns that are entirely null.

    Args:
        cg_df (pd.DataFrame): dataframe of coarse-grained structures
        a_df (pd.DataFrame, optional): dataframe of atomistic structures. Defaults to None. If given, the atomistic dataframe will also be modified to remove the corresponding columns.


    Returns:
        Tuple[pd.DataFrame, pd,DataFrame]: the coarse-grained and atomistic dataframes with the null columns removed
    """
    if a_df is None:
        return cg_df.dropna(axis=1, how="all")
    else:
        return cg_df.dropna(axis=1, how="all"), a_df.dropna(axis=1, how="all")


def concat_dataframes(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of dataframes.

    Args:
        df_list (List[pd.DataFrame]): list of dataframes

    Returns:
        pd.DataFrame: concatenated dataframe
    """
    return pd.concat(df_list, axis=1)


def get_fold_ids(df: pd.DataFrame, numb_folds: int) -> List[np.ndarray]:
    """Separate all the id tags in the dataframe into a specified number of folds.

    Args:
        df (pd.DataFrame): dataframe of structures
        numb_folds (int): number of folds

    Returns:
        List[np.ndarray]: list of arrays; each array contains the id tags for a given fold
    """
    random.seed(42)

    id_tags = list(df.columns)
    random.shuffle(id_tags)

    if numb_folds == 1:
        return id_tags
    else:
        fold_ids = np.array_split(id_tags, numb_folds)
        return fold_ids


def get_energies(
    df: pd.DataFrame,
    id_tags: List[str],
    batches: List[int],
    energy_type: str = "e_local_mofff",
) -> List[List[float]]:
    """Get the local energies for a set of structures.
    The set of structures is specified by the id tags and batches.

    Args:
        df (pd.DataFrame): dataframe of structures
        id_tags (List[str]): list of id tags
        batches (List[int]): list of batches you want to get the energies from, i.e. [1,2,3] = get the energies for structures in batches 1, 2 and 3 for each id tag
        energy_type (str, optional): type of energy you want to get: "e_local_mofff" or "energies_mofff". Defaults to "e_local_mofff".

    Returns:
        List[List[float]]: list of lists of energies; each list contains all the energies for a given id tag.
    """
    energies = []
    # loop over id tags
    # for each id tag, get the structures for the specified batches
    for i in id_tags:
        structures = []
        e = []
        for b in batches:
            structures.append(df.loc[b][i].dropna())

        structures = np.concatenate(structures)

        if len(structures) == 0:
            continue
        else:
            # loop over the structures
            # for each structure, get the local energies for each Si atom
            for s in structures:
                e.append(s.arrays[f"{energy_type}"][s.numbers == 14])

        energies.append(np.concatenate(e))

    return energies


def get_opt_hypers(struct_type: str, linker_type: str) -> Tuple[float, float, float]:
    """Get the optimal SOAP hyperparameters for a given structure type, i.e, A_cg, cg or atomistic, as well as the optimal 'noise' value.
    Optimal SOAP hypers are determined by the grid search results while the optimal noise value is determined by Bayesian optimisation.

    Args:
        struct_type (str): A_cg, cg or atomistic

    Returns:
        Tuple[float,float,float]: the two SOAP hypers - soap_cutoff, sigma - and the noise value
    """
    if linker_type == "CH3":
        df = pd.read_csv(
            root_dir / f"results/hypop/train_15000/{struct_type}_CH3/results.csv"
        )

        # sort by test RMSE and get the best result
        best = df.sort_values(by=["result.av_test_rmse"]).iloc[0]

        soap_cutoff = best["config.soap_cutoff"]
        atom_sigma = best["config.atom_sigma"]
        noise = best["config.noise"]

    elif linker_type == "H":
        df = pd.read_csv(
            grid_search_results / f"{struct_type}_{linker_type}/results.csv"
        )

        # sort by test RMSE and get the best result
        best = df.sort_values(by=["result.av_test_rmse"]).iloc[0]

        soap_cutoff = best["config.cutoff"]
        atom_sigma = best["config.sigma"]
        noise = best["config.noise"]

    elif linker_type == "H_new":
        df = pd.read_csv(new_grid_search_results / f"{struct_type}_H_new/results.csv")

        # sort by test RMSE and get the best result
        best = df.sort_values(by=["result.av_test_rmse"]).iloc[0]

        soap_cutoff = best["config.cutoff"]
        atom_sigma = best["config.sigma"]
        noise = 0.2

    return soap_cutoff, atom_sigma, noise


def build_soap_descriptor(
    struct_type: str, soap_cutoff: float, atom_sigma: float, l_max: int = 8
) -> Descriptor:
    """Build a SOAP descriptor with the specified hyperparameters.

    Args:
        struct_type (str): A_cg, cg or atomistic
        soap_cutoff (float): SOAP cutoff
        atom_sigma (float): atom sigma
        l_max (int, optional): l_max value for the SOAP descriptor. Defaults to 8.

    Returns:
        Descriptor: SOAP descriptor
    """
    start = f"soap n_max=16 l_max={l_max} cutoff={soap_cutoff} atom_sigma={atom_sigma} average=F n_Z=1"

    if struct_type == "cg":
        end = " Z=14 n_species=2 species_Z={14 8}"
    elif struct_type == "A_cg":
        end = " Z=14"
    elif struct_type == "atomistic":
        end = " Z=30 n_species=4 species_Z={30 7 6 1}"
    else:
        raise ValueError("Invalid structure type.")

    return Descriptor(start + end)


def get_opt_soap_descriptor(
    struct_type: str, l_max: int = 8
) -> Tuple[Descriptor, float]:
    """Return the SOAP descriptor with the optimal hyperparameters for a given structure type and specified l_max value.
    Also returns the optimal noise value obtained from Bayesian optimisation.

    Args:
        struct_type (str): A_cg, cg or atomistic
        l_max (int): l_max value for the SOAP descriptor, defaults to 8 (obtained from convergence testing)

    Returns:
        Tuple[Descriptor, float]: the SOAP descriptor and the optimal noise value
    """
    soap_cutoff, atom_sigma, noise = get_opt_hypers(struct_type, linker_type="H")
    desc = build_soap_descriptor(struct_type, soap_cutoff, atom_sigma, l_max)

    return desc, noise


def calc_soap_vectors(
    df: pd.DataFrame,
    id_tags: List[str],
    descriptor: Descriptor,
    batches: List[int],
    B_site: bool = True,
) -> List[List[np.ndarray]]:
    """Get the soap vectors for a given set of structures.
    The set of structures is specified by the id tags and batches.

    Args:
        df (pd.DataFrame): dataframe of structures, can be either cg or atomistic
        id_tags (List[str]): list of id tags
        descriptor (Descriptor): SOAP descriptor
        batches (List[int]): list of batches you want to get the structures from, i.e. [1,2,3] = get the soaps for structures in batches 1, 2 and 3 for each id tag
        B_site (bool, optional): whether to include the B sites (Oxygens) in the soap vectors. Defaults to True. If doing A_cg then set to False.

    Returns:
        List[List[np.ndarray]]: list of lists of soap vectors, each list corresponds to all the soap vectors for a given id tag
    """
    soaps = []
    # loop over id tags
    # for each id tag, get the structures for the specified batches
    for i in id_tags:
        structures = []
        v = []
        for b in batches:
            structures.append(df.loc[b][i].dropna())
        structures = np.concatenate(structures)

        if len(structures) == 0:
            continue
        else:
            # loop over the structures
            for s in structures:
                # remove the B site (Oxygen atoms) if B_site is False
                if B_site == False:
                    s = remove_B_sites(s)
                v.append(descriptor.calc(s)["data"])

        soaps.append(np.concatenate(v))

    return soaps


def remove_B_sites(atoms_object: Atoms) -> Atoms:
    """Remove the B site (Oxygen atoms) from a given structure. Only used if B_site is False, i.e., if doing A_cg.

    Args:
        atoms_object (Atoms): the atoms object for a given structure

    Returns:
        Atoms: the atoms object with the oxygen atoms removed
    """
    silicons = atoms_object.numbers == 14
    atoms_object = atoms_object[silicons]
    return atoms_object


def get_reference_structure(linker_type: str = "H") -> Atoms:
    """Get the reference structure, i.e., the structure with the lowest total energy.
    This should be the ZIF zni structure (id tag = AB2_MOF-128). This structure is used to 'normalise' the energies.

    Args:
        linker_type (str): linker type, either H or CH3. Defaults to H.
    Returns:
        Atoms: the atoms object for the reference structure
    """

    cg_df, _ = get_complete_dataframes(energy_cutoff=1, im_linker=linker_type)

    ls_structures = cg_df.values.reshape(-1)  # convert to 1D array of atoms objects
    ls_structures = ls_structures[~pd.isnull(ls_structures)]  # remove nan values

    zni_structures = [
        s
        for s in ls_structures
        if (s.info["topology"] == "zni" and s.info["RefCode"] == "IMIDZB")
    ]
    ref_struct = [s for s in zni_structures if s.info["batch"] == 1][0]

    return ref_struct


def normalise_energies(energies: np.ndarray, ref: Atoms) -> np.ndarray:
    """Normalise the energies by subtracting the average local energy of the reference structure.

    Args:
        energies (np.ndarray): the energies to be normalised
        ref (Atoms): atoms object for the reference structure

    Returns:
        np.ndarray: the normalised energies
    """

    ref = ref[ref.numbers == 14]
    total_e = np.nansum(ref.arrays["e_local_mofff"])
    numb_Asites = len(ref)

    # average local energy of the reference structure
    av_local_e = total_e / numb_Asites

    # normalise the energies
    normalised_energies = energies - av_local_e

    return normalised_energies


def map_chemical_symbols(structure, mapping):
    original_symbols = structure.get_chemical_symbols()
    new_symbols = [mapping[s] for s in original_symbols]
    structure.set_chemical_symbols(new_symbols)


def get_curated_data(
    struct_type: str, coarse_grain: bool, im_linker: str = "H"
) -> pd.DataFrame:
    """Organises all the structures in the database into a dataframe.
    Rows are indexed according to the following parameters:
    - batch: the batch number, i.e 1, 2, 3, 4, 5 or 6
    - rattling: the rattling level, i.e. 1, 2 or 3
    - gen-code: the way in which the rattled structures were generated i.e. relax then decorate (r-d) or decorate then relax (d-r)
    Each column corresponds to a given structure ID tag.
    Each cell contains a structure as an ase.Atoms object.
    Each atoms object can be identified by its unique combination of batch, rattling, gen-code and id.

                                 |               id tag
    +------+----------+----------+------------+-----------+-----------+
    |batch | rattling | gen-code |  AB1_MOF-1 | AB1_MOF-2 | AB1_MOF-3 |
    +------+----------+----------+------------+-----------+-----------+
    |  1   |   None   | None     |  Atoms     | Atoms     | Atoms     |
    +------+----------+----------+------------+-----------+-----------+
    |      |          | d-r      |  Atoms     | Atoms     | Atoms     |
    |      |   small  +----------+------------+-----------+-----------+
    |      |          | r-d      |  Atoms     | Atoms     | Atoms     |
    |      +----------+----------+------------+-----------+-----------+
    |      |          | d-r      |  Atoms     | Atoms     | Atoms     |
    |  2   |  medium  +----------+------------+-----------+-----------+
    |      |          | r-d      |  Atoms     | Atoms     | Atoms     |
    |      +----------+----------+------------+-----------+-----------+
    |      |          | d-r      |  Atoms     | Atoms     | Atoms     |
    |      |   large  +----------+------------+-----------+-----------+
    |      |          | r-d      |  Atoms     | Atoms     | Atoms     |
    +------+----------+----------+------------+-----------+-----------+

    Args:
        struct_type (str): MOF or Zeolite
        coarse_grain (bool): coarse-grained or atomistic structures
        im_linker (str, optional): the linker type. Defaults to "H". Options are 'H' or 'CH3'.

    Returns:
        pd.Dataframe: all structures organised into a dataframe.
    """
    s_list = []

    if im_linker == "H":
        all_batches = [1, 2, 3, 4, 5, 6]
    else:
        all_batches = [1, 2, 3, 4, 5]

    # loop through all batches
    for b in all_batches:
        # batch 1 is not rattled
        if b == 1:
            file_id = get_file_identifier(
                coarse_grain=coarse_grain, batch_number=b, im_linker=im_linker
            )
            structures = read(
                hZIF_data / f"{struct_type}/MOFFF-curated/{im_linker}/{file_id}.xyz",
                index=":",
            )
            # loop through all structures in batch 1 (i.e. no rattling)
            # label each structure with its batch number, rattling level, gen-code and ID tag
            for s in structures:
                s_list.append(
                    {
                        "batch": b,
                        "rattling": 0,
                        "gen-code": "None",
                        "id": s.info["id"],
                        "structure": s,  # the structure as an ase.Atoms object
                    }
                )
        # batches 2-6 are rattled
        else:
            # loop through the 3 rattling levels
            for r in r_levels:
                r_params = r_levels[r]
                # loop through the 2 gen-codes
                for g in gen_codes:
                    file_id = get_file_identifier(
                        **r_params,
                        gen_code=g,
                        coarse_grain=coarse_grain,
                        batch_number=b,
                        im_linker=im_linker,
                    )
                    structures = read(
                        hZIF_data
                        / f"{struct_type}/MOFFF-curated/{im_linker}/{file_id}.xyz",
                        index=":",
                    )
                    # loop through all structures in the batch
                    # label each structure with its batch number, rattling level, gen-code and ID tag
                    for s in structures:
                        s_list.append(
                            {
                                "batch": b,
                                "rattling": r,
                                "gen-code": g,
                                "id": s.info["id"],
                                "structure": s,
                            }
                        )

    # convert the list of dictionaries into a dataframe
    df = pd.DataFrame(s_list)

    # pivot the dataframe so that each column corresponds to a given structure ID tag
    df = df.pivot(
        index=["batch", "rattling", "gen-code"],
        columns="id",
        values="structure",
    )
    return df
