import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

import gdown # type: ignore
import zipfile
from typing import Any

class GWDataset(Dataset):
    def __init__(self, root: str, split: str, download: bool = False, cleanup: bool = True) -> None:
        self.root = root
        if not os.path.exists(os.path.join(self.root, "dataset")):
            raise RuntimeError(f"No Data found at {os.path.abspath(self.root)}")

        self.split = split
        self.download = download

        if self.download:
            file_url = "https://drive.google.com/file/d/1CIi0WDelwFhWDFlCpRe0M9VJJUUKUg7n"
            output_path = os.path.join(self.root, 'dataset.zip')

            gdown.download(file_url, output_path, quiet=False)
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)

            if cleanup: os.remove(os.path.join(self.root, "dataset.zip"))

        self.split_df = self._prepare_split()

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
            return df.iloc[indices]
        
        if self.split == "val":
            indices = np.concatenate([noise_val_indices, bbh_val_indices, bns_val_indices])
            indices.sort()
            return df.iloc[indices]
        
        if self.split == "test":
            indices = np.concatenate([noise_test_indices, bbh_test_indices, bns_test_indices])
            indices.sort()
            return df.iloc[indices]
        
        raise RuntimeError("Invalid Split")

    def __len__(self) -> int:
        return len(self.split_df)
        
    def __getitem__(self, index: Any) -> Any:
        raise NotImplementedError("`__getitem__` not implemented for the chosen dataset")
