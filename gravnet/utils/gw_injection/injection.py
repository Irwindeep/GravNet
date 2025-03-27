import numpy as np
from scipy.signal import welch # type: ignore
from pycbc.types import TimeSeries # type: ignore

from numpy.typing import NDArray
from typing import Tuple

SEG_LEN = 4096
SAMPLING_FREQ = 1/4096

def inject_waveform(
    noise: NDArray[np.float64], waveform: TimeSeries,
    snr: float
) -> Tuple[NDArray[np.float64], float]:
    frequencies, psd_strain = welch(noise, SAMPLING_FREQ, nperseg=SEG_LEN)
    
    N = len(waveform)
    H_f = np.fft.rfft(waveform)

    freq_template = np.fft.rfftfreq(N, 1/SAMPLING_FREQ)

    psd_interp = np.interp(freq_template, frequencies, psd_strain)

    df = freq_template[1] - freq_template[0]
    integrand = (np.abs(H_f)**2) / psd_interp
    current_snr = np.sqrt(4.0 * np.sum(integrand) * df)

    scaling_factor = snr/current_snr
    scaled_waveform = scaling_factor * waveform
    
    gw_scaled = noise + scaled_waveform.data
    gw_scaled = np.fft.irfft(np.fft.rfft(gw_scaled)/psd_interp**0.5).real

    return gw_scaled, scaling_factor
