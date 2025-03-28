import os
import numpy as np
from scipy.signal import welch # type: ignore
import torch

from pycbc.types import TimeSeries # type: ignore
from .gw_dataset import GWDataset

from gravnet.utils.gw_injection import inject_waveform
from numpy.typing import NDArray
from typing import Tuple

class NoiseSegData(GWDataset):
    def __init__(self, root, split, download = False, cleanup = True) -> None:
        super().__init__(root, split, download, cleanup)
    
    def _prepare_synth_data(self, noise: NDArray[np.float64], waveform: NDArray[np.float64], snr: float) -> torch.Tensor:
        injected_waveform, _ = inject_waveform(
            noise, TimeSeries(waveform, delta_t=1/4096, epoch=0), snr
        )
        return torch.tensor(injected_waveform, dtype=torch.float32)
    
    def _getitem(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.split_df.iloc[index]

        noise_file = os.path.join(self.root, "dataset", "gwaves", row["data_file"])
        noise = np.load(noise_file)
        if row["category"] == 0:
            frequencies, psd_strain = welch(noise, 1/4096, nperseg=4096)
            freq_template = np.fft.rfftfreq(4096, 4096)
            psd_interp = np.interp(freq_template, frequencies, psd_strain)
            noise = np.fft.irfft(np.fft.rfft(noise)/psd_interp**0.5).real
            signal = torch.tensor(noise, dtype=torch.float32)
        else:
            waveform_file = os.path.join(self.root, "dataset", "simulations", row["data_file"])
            waveform = np.load(waveform_file)
            signal = self._prepare_synth_data(noise, waveform, row["snr"])

            frequencies, psd_strain = welch(noise, 1/4096, nperseg=4096)
            freq_template = np.fft.rfftfreq(4096, 4096)
            psd_interp = np.interp(freq_template, frequencies, psd_strain)
            noise = np.fft.irfft(np.fft.rfft(noise)/psd_interp**0.5).real

        return signal, torch.tensor(noise, dtype=torch.float32)
