import torch

import pandas as pd

import os
import shutil
from pathlib import Path
import random
import math


numbers = ["3", "11", "16", "9", "0"]
paths = []

df_new = pd.DataFrame()
for i in numbers:
    df = pd.read_csv(
        "/Users/keshav/Developer/pytorch-learning/Custom Datasets/archive/train_df.csv",
        index_col=False,
    )

    df["Target"] = df["Target"].str.strip()
    df = df[df.Target == i]
    filtered_rows = df[df.Target == i].copy()

    df_new = df_new.append(filtered_rows)

# print(len(df_new))
df_new = df_new.reset_index(drop=True)
df_new.to_csv(
    "/Users/keshav/Developer/pytorch-learning/Custom Datasets/archive/images/data/subset.csv"
)

TRAIN_IMAGE_ID = []
TEST_IMAGE_ID = []
TRAIN_IMAGE_PATH = []
TEST_IMAGE_PATH = []

TRAIN_TARGET = []
TEST_TARGET = []
for i in numbers:

    DATA_PATH = Path(
        "/Users/keshav/Developer/pytorch-learning/Custom Datasets/archive/images/data"
    )
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(DATA_PATH / f"train/{i}"):
        os.makedirs(DATA_PATH / f"train/{i}")
        if not os.path.exists(DATA_PATH / f"test/{i}"):
            os.makedirs(DATA_PATH / f"test/{i}")
        else:
            print("Folder already exists.")
        df = pd.read_csv(
            "/Users/keshav/Developer/pytorch-learning/Custom Datasets/archive/train_df.csv",
            index_col=False,
        )

        df["Target"] = df["Target"].str.strip()
        df = df[(df.Target == f"{i}")]

        df["image_path"] = df["image_path"].str.replace(
            "./images",
            "/Users/keshav/Developer/pytorch-learning/Custom Datasets/archive/images",
            regex=True,
        )

        paths = df.image_path.tolist()

        paths = [f"{item}" for item in paths]

        train_paths = random.sample(paths, k=(math.ceil(len(paths) * 0.8)))

        test_paths = [elem for elem in paths if elem not in train_paths]

        for j in train_paths:
            a = j.replace(
                "/Users/keshav/Developer/pytorch-learning/Custom Datasets/archive/images/train",
                "/Users/keshav/Developer/pytorch-learning/Custom Datasets/archive/images/train",
            )
            TRAIN_IMAGE_PATH.append(j)

            TRAIN_IMAGE_ID.append(
                os.path.splitext(os.path.basename(a))[0].replace("-c", "")
            )

            shutil.copy(a, DATA_PATH / f"train/{i}")
        for k in test_paths:
            b = k.replace(
                "/Users/keshav/Developer/pytorch-learning/Custom Datasets/archive/images/train",
                "/Users/keshav/Developer/pytorch-learning/Custom Datasets/archive/images/train",
            )
            TEST_IMAGE_PATH.append(b)

            TEST_IMAGE_ID.append(
                os.path.splitext(os.path.basename(b))[0].replace("-c", "")
            )

            shutil.copy(b, DATA_PATH / f"test/{i}")

    else:
        print("Folder already exists.")


for i in range(len(TRAIN_IMAGE_ID)):
    TRAIN_TARGET.append(
        (df_new[df_new["SOPInstanceUID"] == TRAIN_IMAGE_ID[i]]["Target"].item())
    )
    TRAIN_TARGET = [str(x) for x in TRAIN_TARGET]
for i in range(len(TEST_IMAGE_ID)):
    TEST_TARGET.append(
        (df_new[df_new["SOPInstanceUID"] == TEST_IMAGE_ID[i]]["Target"].item())
    )
    TEST_TARGET = [str(x) for x in TEST_TARGET]


train_df = pd.DataFrame()
test_df = pd.DataFrame()
train_df["SOPInstanceUID"] = TRAIN_IMAGE_ID
train_df["Target"] = TRAIN_TARGET

train_df["image_path"] = TRAIN_IMAGE_PATH
train_df.to_csv(
    "/Users/keshav/Developer/pytorch-learning/Custom Datasets/archive/images/data/train_subset.csv"
)
test_df["SOPInstanceUID"] = TEST_IMAGE_ID
test_df["Target"] = TEST_TARGET

test_df["image_path"] = TEST_IMAGE_PATH
test_df.to_csv(
    "/Users/keshav/Developer/pytorch-learning/Custom Datasets/archive/images/data/test_subset.csv"
)


train_dir = DATA_PATH / "train"
test_dir = DATA_PATH / "test"
