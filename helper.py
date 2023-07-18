import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from typing import Tuple


def get_config(file="config.yaml"):
    with open(file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def get_partial_df(
    root: str, seed=18
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Get training and validation dataframe for driver_161_90 files only

    Args:
        root (str): root directory
        seed (int, optional): Random seed. Defaults to 18.

    Returns:
        tuple: pandas Series of train and train ground truth directories, valid and valid ground truth directories,
    """
    config = get_config()
    split_size = float(config["split_size"])

  
    df = pd.read_csv(
        os.path.join(root, "data/list/train_gt.txt"), delim_whitespace=True, header=None
    )
        
    df = df[df[0].str.contains("driver_161_90")].reset_index(drop=True)
    df[0] = df[0].replace(
        {"/driver_161_90frame": os.path.join(root, "data/driver_161_90frame")},
        regex=True,
    )
    df[1] = df[1].replace(
        {
            "/laneseg_label_w16/driver_161_90frame": os.path.join(
                root, "data/laneseg_label_w16/driver_161_90frame"
            )
        },
        regex=True,
    )

    train_df, valid_df = train_test_split(df, test_size=split_size, random_state=seed)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    train_dir = train_df[0].copy()
    train_dir_gt = train_df[1].copy()

    valid_dir = valid_df[0].copy()
    valid_dir_gt = valid_df[1].copy()

    return train_dir, train_dir_gt, valid_dir, valid_dir_gt


def get_df(root: str, seed=18) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Get training and validation dataframe for all training data of CULane

    Args:
        root (str): root directory
        seed (int, optional): Random seed. Defaults to 18.

    Returns:
        tuple: pandas Series of train and train ground truth directories, valid and valid ground truth directories,
    """
    config = get_config()
    split_size = float(config["split_size"])

    df = pd.read_csv(
        os.path.join(root, "data/list/train_gt.txt"), delim_whitespace=True, header=None
    )

    df[0] = df[0].replace({"/driver": os.path.join(root, "data/driver")}, regex=True)
    df[1] = df[1].replace(
        {"/laneseg_label": os.path.join(root, "data/laneseg_label")}, regex=True
    )

    train_df, valid_df = train_test_split(df, test_size=split_size, random_state=seed)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    train_dir = train_df[0].copy()
    train_dir_gt = train_df[1].copy()

    valid_dir = valid_df[0].copy()
    valid_dir_gt = valid_df[1].copy()

    return train_dir, train_dir_gt, valid_dir, valid_dir_gt
