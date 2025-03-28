import os
import numpy as np
import torch

from pycbc.types import TimeSeries #type: ignore
from .gw_dataset import GWDataset

from gravnet.utils.gw_injection import inject_waveform
from numpy.typing import NDArray
from typing import Tuple

class ParamEstData(GWDataset):
    def __init__(self, root, split, download = False, cleanup = True) -> None:
        super().__init__(root, split, download, cleanup)
        self.split_df = self.split_df[self.split_df["category"] > 0]

    def __len__(self) -> int:
        return len(self.split_df)
    
    def _prepare_synth_data(self, noise: NDArray[np.float64], waveform: NDArray[np.float64], snr: float) -> torch.Tensor:
        injected_waveform, _ = inject_waveform(
            noise, TimeSeries(waveform, delta_t=1/4096, epoch=0), snr
        )
        return torch.tensor(injected_waveform, dtype=torch.float32)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.split_df.iloc[index]

        noise_file = os.path.join(self.root, "dataset", "gwaves", row["data_file"])
        waveform_file = os.path.join(self.root, "dataset", "simulations", row["data_file"])
        m1, m2, snr = row["m1"], row["m2"], row["snr"]
        
        noise = np.load(noise_file)
        waveform = np.load(waveform_file)
        injected_signal = self._prepare_synth_data(noise, waveform, snr)
        params = torch.tensor([m1, m2, snr], dtype=torch.float32)

        return injected_signal, params
