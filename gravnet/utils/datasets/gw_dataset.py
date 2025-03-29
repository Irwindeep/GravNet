import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

import gdown # type: ignore
import zipfile
from typing import Any

class GWDataset(Dataset):
    def __init__(self, root: str, split: str, download: bool = False, cleanup: bool = True) -> None:
        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.split = split
        self.download = download

        if self.download:
            os.makedirs(f"{self.root}/dataset", exist_ok=True)
            self._load_dataset(cleanup)
            
            os.makedirs(f"{self.root}/dataset/simulations", exist_ok=True)
            self._load_simulations(cleanup)

        self.split_df = self._prepare_split()
        self._cache_dir = os.path.join(self.root, "cache", self.split)
        os.makedirs(self._cache_dir, exist_ok=True)

    def _load_dataset(self, cleanup: bool):
        already_exists = (
            os.path.exists(os.path.join(self.root, "dataset", "gwaves")) and
            len(os.listdir(os.path.join(self.root, "dataset", "gwaves"))) == 256379
        )
        if already_exists: print("Files aleady downloaded and verified")
        else:
            file_url = "https://drive.google.com/uc?id=1lqDJOeLvCrPR34QXVZzN4pMIz8RQPmqy"
            output_path = os.path.join(self.root, "dataset.zip")

            gdown.download(file_url, output_path, quiet=False)
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)

            if cleanup: os.remove(os.path.join(self.root, "dataset.zip"))

    def _load_simulations(self, cleanup):
        already_exists = (
            os.path.exists(os.path.join(self.root, "dataset", "simulations")) and
            len(os.listdir(os.path.join(self.root, "dataset", "simulations"))) == 120815
        )
        if already_exists: print("Files aleady downloaded and verified")
        else:
            file_url = "https://drive.google.com/uc?id=1pVXt-qDxMUeTxi8M789VjIe8DBqEZJTe"
            output_path = os.path.join(self.root, "dataset", "simulations.zip")

            gdown.download(file_url, output_path, quiet=False)
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.root, "dataset"))

            if cleanup: os.remove(os.path.join(self.root, "dataset", "simulations.zip"))

    def _prepare_split(self) -> pd.DataFrame:
        np.random.seed(12)
        df = pd.read_csv(os.path.join(self.root, "dataset", "gwaves_attr.csv"))

        noise_indices = df[df["category"] == 0].index.to_numpy()
        bbh_indices = df[df["category"] == 1].index.to_numpy()
        bns_indices = df[df["category"] == 2].index.to_numpy()

        noise_train_val_indices = np.random.choice(
            noise_indices, size=len(noise_indices)//2, replace=False
        )
        noise_train_indices = np.random.choice(
            noise_train_val_indices, size=int(0.8*len(noise_train_val_indices)), replace=False
        )
        noise_val_indices = np.setdiff1d(noise_train_val_indices, noise_train_indices)
        noise_test_indices = np.setdiff1d(noise_indices, noise_train_val_indices)

        bbh_train_val_indices = np.random.choice(
            bbh_indices, size=len(bbh_indices)//2, replace=False
        )
        bbh_train_indices = np.random.choice(
            bbh_train_val_indices, size=int(0.8*len(bbh_train_val_indices)), replace=False
        )
        bbh_val_indices = np.setdiff1d(bbh_train_val_indices, bbh_train_indices)
        bbh_test_indices = np.setdiff1d(bbh_indices, bbh_train_val_indices)

        bns_train_val_indices = np.random.choice(
            bns_indices, size=len(bns_indices)//2, replace=False
        )
        bns_train_indices = np.random.choice(
            bns_train_val_indices, size=int(0.8*len(bns_train_val_indices)), replace=False
        )
        bns_val_indices = np.setdiff1d(bns_train_val_indices, bns_train_indices)
        bns_test_indices = np.setdiff1d(bns_indices, bns_train_val_indices)

        if self.split == "train":
            indices = np.concatenate([noise_train_indices, bbh_train_indices, bns_train_indices])
            indices.sort()
            return df.iloc[indices].reset_index()
        
        if self.split == "val":
            indices = np.concatenate([noise_val_indices, bbh_val_indices, bns_val_indices])
            indices.sort()
            return df.iloc[indices].reset_index()
        
        if self.split == "test":
            indices = np.concatenate([noise_test_indices, bbh_test_indices, bns_test_indices])
            indices.sort()
            return df.iloc[indices].reset_index()
        
        raise RuntimeError("Invalid Split")

    def __len__(self) -> int:
        return len(self.split_df)
    
    def __getitem__(self, index: Any) -> Any:
        cache_file = os.path.join(self._cache_dir, f"{index}.pt")
        if os.path.exists(cache_file):
            return torch.load(cache_file)
        
        item = self._getitem(index)
        torch.save(item, cache_file)
        return item

    def _getitem(self, index: Any) -> Any:
        raise NotImplementedError("`_getitem` not implemented for the chosen dataset")
