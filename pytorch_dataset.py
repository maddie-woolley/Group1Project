# Provided by GeoLifeClef
# https://github.com/maximiliense/GLC/blob/master/data_loading/pytorch_dataset.py

from __future__ import annotations

import collections
import itertools
from pathlib import Path
from typing import Callable, Optional, Union, TYPE_CHECKING

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from common import load_patch

if TYPE_CHECKING:
    import numpy.typing as npt
    from common import Patches
    from environmental_raster import PatchExtractor

    Targets = npt.NDArray[np.int64]


class GeoLifeCLEF2022Dataset(Dataset):
    """Pytorch dataset handler for GeoLifeCLEF 2022 dataset.
    Parameters
    ----------
    root : string or pathlib.Path
        Root directory of dataset.
    subset : string, either "train", "val", "train+val" or "test"
        Use the given subset ("train+val" is the complete training data).
    region : string, either "both", "fr" or "us"
        Load the observations of both France and US or only a single region.
    patch_data : string or list of string
        Specifies what type of patch data to load, possible values: 'all', 'rgb', 'near_ir', 'landcover' or 'altitude'.
    use_rasters : boolean (optional)
        If True, extracts patches from environmental rasters.
    patch_extractor : PatchExtractor object (optional)
        Patch extractor to use if rasters are used.
    transform : callable (optional)
        A function/transform that takes a list of arrays and returns a transformed version.
    target_transform : callable (optional)
        A function/transform that takes in the target and transforms it.
    """

    def __init__(
            self,
            root,
            subset,
            *,
            region="both",
            patch_data="all",
            use_rasters=True,
            patch_extractor=None,
            transform=None,
            target_transform=None
    ):
        self.root = Path(root)
        self.subset = subset
        self.region = region
        self.patch_data = patch_data
        self.transform = transform
        self.target_transform = target_transform

        possible_subsets = ["train", "val", "train+val", "test"]
        if subset not in possible_subsets:
            raise ValueError(
                "Possible values for 'subset' are: {} (given {})".format(
                    possible_subsets, subset
                )
            )

        possible_regions = ["both", "fr", "us"]
        if region not in possible_regions:
            raise ValueError(
                "Possible values for 'region' are: {} (given {})".format(
                    possible_regions, region
                )
            )

        if subset == "test":
            subset_file_suffix = "test"
            self.training_data = False
        else:
            subset_file_suffix = "train"
            self.training_data = True

        # df_fr = pd.read_csv(
        #     self.root
        #     / "observations"
        #     / "observations_us_{}.csv".format(subset_file_suffix),  # FIXME to France later if needed
        #     sep=";",
        #     index_col="observation_id", nrows=50000
        # )
        df_us = pd.read_csv(
            self.root
            / "observations"
            / "observations_us_{}.csv".format(subset_file_suffix),
            index_col="observation_id", nrows=10000
        )

        # if region == "both":
        #     df = pd.concat((df_fr, df_us))
        # elif region == "fr":
        #     df = df_fr
        # elif region == "us":
        #     df = df_us
        if region == "us":
            df = df_us

        if self.training_data and subset != "train+val":
            ind = df.index[df["subset"] == subset]
            df = df.loc[ind]

        self.observation_ids = df.index
        self.coordinates = df[["latitude", "longitude"]].values

        if self.training_data:
            # torch.tensor(df["species_id"].values, dtype = torch.long)
            self.targets = df['genus_id'].values
        else:
            self.targets = None

        # FIXME: add back landcover one hot encoding?
        # self.one_hot_size = 34
        # self.one_hot = np.eye(self.one_hot_size)

        if use_rasters:
            if patch_extractor is None:
                # from .environmental_raster import PatchExtractor

                patch_extractor = PatchExtractor(self.root / "rasters", size=256)
                patch_extractor.add_all_rasters()

            self.patch_extractor = patch_extractor
        else:
            self.patch_extractor = None

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, index):
        latitude = self.coordinates[index][0]
        longitude = self.coordinates[index][1]
        observation_id = self.observation_ids[index]
        # print(observation_id)
        try:
            patches = load_patch(
                observation_id, self.root, data=self.patch_data
            )

        except ValueError:
            pass

        patches = torch.Tensor(patches)
        # FIXME: add back landcover one hot encoding?
        # lc = patches[3]
        # lc_one_hot = np.zeros((self.one_hot_size,lc.shape[0], lc.shape[1]))
        # row_index = np.arange(lc.shape[0]).reshape(lc.shape[0], 1)
        # col_index = np.tile(np.arange(lc.shape[1]), (lc.shape[0], 1))
        # lc_one_hot[lc, row_index, col_index] = 1

        # Extracting patch from rasters
        if self.patch_extractor is not None:
            # this will have all the bioclimatic or pedologic rasters for the specific lat, long position
            #print(observation_id, latitude, longitude)
            environmental_patches = self.patch_extractor[(latitude, longitude)]
            # patches = patches + torch.from_numpy(np.array(environmental_patches))
            # convert list to pytorch tensor
            # print (patches[0].size, patches[1].size, patches[2].size, patches[3].size)   #196608 65536 65536 65536
            # patches =  tf.ragged.constant(patches)
            # convert numpy to pytorch tensor
            # environmental_patches = torch.from_numpy(environmental_patches)
            # print (patches.shape)
            # print (environmental_patches.shape)  # 20,256,256
            patches = patches + torch.Tensor(environmental_patches)

        # Concatenate all patches into a single tensor
        if len(patches) == 1:
            patches = patches[0]

        if self.transform:
            patches = self.transform(patches)
            # patches = self.transform(image=patches)["image"]
            # print (patches.shape)

        if self.training_data:
            target = self.targets[index]

            if self.target_transform:
                target = self.target_transform(target)

            return patches, target
        else:
            return patches
