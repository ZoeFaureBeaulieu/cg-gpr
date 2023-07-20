from math import log
import numpy as np
import pandas as pd
from typing import List, Tuple
import pyGPs
from quippy.descriptors import Descriptor
from data import get_energies, calc_soap_vectors
import random


def gpr_with_cv(
    noise: int,
    df: pd.DataFrame,
    fold_ids: List[np.ndarray],
    train_batches: List[int],
    test_batches: List[int],
    descriptor: Descriptor,
    atomistic_df: pd.DataFrame = None,
    numb_train: int = None,
    B_site: bool = True,
    energy_type: str = "e_local_mofff",
) -> Tuple[float, float, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Perform cross-validation on a Gaussian Process Regression model.

    Args:
        model (pyGPs.GPR): The model to perform cross-validation on.
        df (pd.DataFrame): The dataframe containing all the coarse-grained structures
        fold_ids (List[np.ndarray[str]]): List of k arrays. Each array contains the id tags for the structures to be included in that fold
        train_batches (List[int]): The batches to use for training
        test_batches (List[int]): The batches to use for testing
        descriptor (Descriptor): The SOAP descriptor
        atomistic_df (pd.DataFrame, optional): The dataframe containing all the atomistic structures, by default None. If not None, the atomistic SOAPs will be used for training.
        numb_train (int, optional): The number of training environments to use, by default None. If None, all environments will be used.
        B_site (bool, optional): whether to include the B sites (Oxygens) in the soap vectors. Defaults to True. If doing A_cg then set to False.
        energy_type (str, optional): The energy type to use:  "e_local_mofff" or "energies_mofff". Defaults to "e_local_mofff" which takes into account the energies of neighbouring Im- rings.

    Returns:
        Tuple[float,float,list[np.ndarray],list[np.ndarray], List[np.ndarray]]: train error, test error (both averaged over all folds); train set predictions, test set predictions, test set labels.
    """

    k = len(fold_ids)

    train_rmses = []
    test_rmses = []
    all_train_predictions = []
    all_test_predictions = []
    test_labels = []

    # loop over k folds
    for i in range(k):
        print(f"Starting fold {i}")

        # get the training and testing structure id tags
        train_tags = np.concatenate([fold_ids[j] for j in range(k) if j != i])
        test_tags = fold_ids[i]

        error_message = "Train and test id tags overlap."
        assert all([j not in train_tags for j in test_tags]), error_message

        def get_soaps_and_energies(
            id_tags: List[str], batches: List[int]
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Returns the soap vectors and energies (labels) for the given set of structures.
            The set of structures is defined by the id tags and batches.

            Args:
                id_tags (List[str]): list of id tags
                batches (List[int]): list of batches

            Returns:
                Tuple[np.ndarray, np.ndarray]: SOAP vectors and energies
            """

            energies = get_energies(df, id_tags, batches, energy_type=energy_type)
            energies = np.concatenate(energies).reshape(-1, 1)  # reshape to 2D array

            # if atomistic_df is not None, use the atomistic SOAPs for training
            if atomistic_df is not None:
                soaps = calc_soap_vectors(
                    atomistic_df, id_tags, descriptor, batches, B_site=B_site
                )
            # else use the coarse-grained SOAPs
            else:
                soaps = calc_soap_vectors(
                    df, id_tags, descriptor, batches, B_site=B_site
                )

            soaps = np.concatenate(soaps)
            error_message = "Number of energies and SOAPs do not match"
            assert len(energies) == len(soaps), error_message

            return soaps, energies

        # get the training and testing SOAPs and energies
        train_soaps, train_energies = get_soaps_and_energies(train_tags, train_batches)
        test_soaps, test_energies = get_soaps_and_energies(test_tags, test_batches)
        test_labels.append(test_energies)

        # if specified, select a subset of the complete training data
        if numb_train is not None:
            random.seed(42)
            list = random.sample(range(0, len(train_energies)), numb_train)
            train_energies = train_energies[list]
            train_soaps = train_soaps[list]

        print(f"Number of training points: {len(train_energies)}")
        # train the model
        model = train_gpr(noise, train_soaps, train_energies)

        # get the training set predictions and errors
        train_predictions, train_error = predict_gpr(model, train_soaps, train_energies)
        train_rmses.append(train_error)
        all_train_predictions.append(train_predictions)

        # get the test set predictions and errors
        test_predictions, test_error = predict_gpr(model, test_soaps, test_energies)
        test_rmses.append(test_error)
        all_test_predictions.append(test_predictions)

        print(f"Train RMSE: {train_error:.3} Test RMSE: {test_error:.3}")

    return (
        np.mean(train_rmses),
        np.mean(test_rmses),
        all_train_predictions,
        all_test_predictions,
        test_labels,
    )


def train_gpr(noise: float, X_train: np.ndarray, y_train: np.ndarray) -> pyGPs.GPR:
    """GPR model used for training.

    Args:
        noise (float): the regulariser value in eV
        X_train (np.ndarray): the training data. e.g. SOAP vectors of the training structures
        y_train (np.ndarray): the training set labels. e.g. local energies of the atoms in the training structures

    Returns:
        PyGPs.GPR: the trained model
    """
    # initialise the model
    model = pyGPs.GPR()
    model.setNoise(log(noise))
    model.setPrior(kernel=pyGPs.cov.Poly(d=4), mean=pyGPs.mean.Const(y_train.mean()))

    # train the model
    model.getPosterior(X_train, y_train)

    return model


def predict_gpr(
    model: pyGPs.GPR, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Obtains the predictions and RMSE of the model on test data.

    Args:
        model (pyGPs.GPR): the trained model
        X_test (np.ndarray): the testing data. e.g. SOAP vectors of the testing structures
        y_test (np.ndarray): the testing set labels. e.g. local energies of the atoms in the testing structures

    Returns:
        Tuple[np.ndarray, float]: test predictions and test error
    """
    # get the predictions
    predictions = model.predict(X_test)[0]
    # get the RMSE
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

    return predictions, rmse
