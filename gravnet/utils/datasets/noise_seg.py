import os
import numpy as np
from scipy.signal import welch
import torch
from .gw_dataset import GWDataset

from gravnet.utils.gw_injection import (
    generate_simulated_waveform,
    adjust_simulated_waveform,
    inject_waveform
)
from numpy.typing import NDArray
from typing import Tuple

class NoiseSegData(GWDataset):
    def __init__(self, root, split, download = False, cleanup = True) -> None:
        super().__init__(root, split, download, cleanup)
    
    def _prepare_synth_data(
        self, noise: NDArray[np.float64], approximant: str, m1: float, m2: float,
        snr: float
    ) -> torch.Tensor:
        waveform = generate_simulated_waveform(approximant, m1, m2, f_lower=20, delta_t=1/4096)
        waveform = adjust_simulated_waveform(waveform, desired_length=1.0)

        injected_waveform, _ = inject_waveform(noise, waveform, snr)
        return torch.tensor(injected_waveform, dtype=torch.float32)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
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
            m1, m2, snr = row["m1"], row["m2"], row["snr"]
            approximant = "SEOBNRv4" if row["category"] == 1 else "TaylorF2"

            signal = self._prepare_synth_data(noise, approximant, m1, m2, snr)
            frequencies, psd_strain = welch(noise, 1/4096, nperseg=4096)
            freq_template = np.fft.rfftfreq(4096, 4096)
            psd_interp = np.interp(freq_template, frequencies, psd_strain)
            noise = np.fft.irfft(np.fft.rfft(noise)/psd_interp**0.5).real

        return signal, torch.tensor(noise, dtype=torch.float32)
