import unittest
import numpy as np
from scipy.signal import welch # type: ignore

from pycbc.filter import matched_filter # type: ignore
from gwpy.timeseries import TimeSeries # type: ignore
from gravnet.utils.gw_injection import (
    generate_simulated_waveform,
    inject_waveform,
    adjust_simulated_waveform
)

class TestBBHInj(unittest.TestCase):
    def setUp(self):
        approximant = "SEOBNRv4"
        m1, m2 = 57.0, 33.0
        delta_t, desired_length = 1./4096, 1.
        f_lower = 20
        snr = 7.5

        self.noise = np.load("gw_data/dataset/gwaves/L1-1175789870_1175805747_0.npy")
        self.bbh_waveform = generate_simulated_waveform(approximant, m1, m2, f_lower, delta_t)
        self.bbh_waveform = adjust_simulated_waveform(self.bbh_waveform, desired_length)
        self.injected_signal, self.scaling_factor = inject_waveform(self.noise, self.bbh_waveform, snr)

    def test_inj_duration(self):
        assert(self.injected_signal.shape == (4096,))

    def test_inj_snr(self):
        fs = 1/4096
        freqs, noise_psd = welch(self.noise, fs, nperseg=4096)
        freq_template = np.fft.rfftfreq(4096, 4096)
        psd_interp = np.interp(freq_template, freqs, noise_psd)

        waveform = np.fft.irfft(np.fft.rfft(self.bbh_waveform.data)/psd_interp**0.5).real
        snr = matched_filter(
            TimeSeries(waveform, dt=1/4096, t0=0).to_pycbc(),
            TimeSeries(self.injected_signal, dt=1/4096, t0=0).to_pycbc(),
        )

        np.testing.assert_almost_equal(np.max(np.abs(snr.data)), 7.5, decimal=0)


class TestBNSInj(unittest.TestCase):
    def setUp(self):
        approximant = "TaylorF2"
        m1, m2 = 1.4, 1.2
        delta_t, desired_length = 1./4096, 1.
        f_lower = 20
        snr = 7.5

        self.noise = np.load("gw_data/dataset/gwaves/L1-1175789870_1175805747_0.npy")
        self.bns_waveform = generate_simulated_waveform(approximant, m1, m2, f_lower, delta_t)
        self.bns_waveform = adjust_simulated_waveform(self.bns_waveform, desired_length)
        self.injected_signal, self.scaling_factor = inject_waveform(self.noise, self.bns_waveform, snr)

    def test_inj_duration(self):
        assert(self.injected_signal.shape == (4096,))

    def test_inj_snr(self):
        fs = 1/4096
        freqs, noise_psd = welch(self.noise, fs, nperseg=4096)
        freq_template = np.fft.rfftfreq(4096, 4096)
        psd_interp = np.interp(freq_template, freqs, noise_psd)

        waveform = np.fft.irfft(np.fft.rfft(self.bns_waveform.data)/psd_interp**0.5).real
        snr = matched_filter(
            TimeSeries(waveform, dt=1/4096, t0=0).to_pycbc(),
            TimeSeries(self.injected_signal, dt=1/4096, t0=0).to_pycbc(),
        )

        np.testing.assert_almost_equal(np.max(np.abs(snr.data)), 7.5, decimal=0)
