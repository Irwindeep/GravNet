import numpy as np
from gwpy.timeseries import TimeSeries as gwpyTimeSeries
from pycbc.types import TimeSeries as pycbcTimeSeries # type: ignore
from pycbc.filter import matchedfilter # type: ignore
from pycbc.psd import interpolate
from .simulation import adjust_simulated_waveform

from numpy.typing import NDArray
from typing import Tuple

SEG_LEN = 4096
SEG_STRIDE = 2048

def inject_waveform(
    noise: NDArray[np.float64], waveform: pycbcTimeSeries,
    snr: float, desired_length: float
) -> Tuple[NDArray[np.float64], float]:
    noise = gwpyTimeSeries(noise, t0=0, dt=waveform.delta_t)

    waveform_len = len(waveform.data)
    waveform.resize(4096)

    noise_psd = noise.psd().to_pycbc()
    noise_psd = interpolate(noise_psd, delta_f=waveform.delta_f)

    current_snr = matchedfilter.sigma(waveform, noise_psd, low_frequency_cutoff=20)
    scaling_factor = snr/current_snr

    scaled_waveform = scaling_factor * waveform
    scaled_waveform.resize(waveform_len)
    adjusted_waveform = adjust_simulated_waveform(scaled_waveform, desired_length=desired_length)
    
    gw_scaled = noise.value + adjusted_waveform.data
    return gw_scaled, scaling_factor
