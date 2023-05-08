#!/usr/bin/env python3
import sys
from contextlib import nullcontext
from pathlib import Path
import pandas as pd
from filelock import FileLock


def get_df(path: Path) -> pd.DataFrame:
    """Read a csv file into a pandas DataFrame.

    Args:
        path (Path): path to the csv file

    Returns:
        pd.DataFrame: the DataFrame
    """
    try:
        return pd.read_csv(path)
    # if the file is empty or can't be read, return an empty DataFrame
    except:
        return pd.DataFrame()


def merge(a: str, b: str, c: str) -> None:
    """Merge two csv files,a and b, into a third csv file, c.

    Args:
        a (str): the first csv file path
        b (str): the second csv file path
        c (str): the merged csv file path (can be the same as a or b)
    """
    # convert the paths to Path objects
    a = Path(a)
    b = Path(b)
    c = Path(c)

    # create file locks for a and b
    locka = FileLock(a.with_suffix(".lock"))
    lockb = FileLock(b.with_suffix(".lock"))

    # if the output file c is not a or b, create a file lock for c
    if c != a and c != b:
        lockc = FileLock(c.with_suffix(".lock"))
    else:
        lockc = nullcontext()

    # lock the files
    # this will prevent other processes from reading or writing to the files
    with locka, lockb, lockc:
        # read the csv files into DataFrames
        dfa = get_df(a)
        dfb = get_df(b)

        # merge the DataFrames and remove duplicates
        dfc = pd.concat([dfa, dfb]).drop_duplicates()

        # write the merged DataFrame to the output file c
        dfc.to_csv(c, index=False)

    # remove the file locks, if they exist, if not, do nothing
    a.with_suffix(".lock").unlink(missing_ok=True)
    b.with_suffix(".lock").unlink(missing_ok=True)
    c.with_suffix(".lock").unlink(missing_ok=True)


if __name__ == "__main__":
    # check that the correct number of arguments have been passed
    assert len(sys.argv) == 4, "Usage: ./merge_csv.py <a> <b> <c>"

    # assign the first three arguments to a, b and c
    a, b, c = sys.argv[1:]

    # merge the csv files using the function defined above
    merge(a, b, c)
