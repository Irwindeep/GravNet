import unittest
import numpy as np
from gravnet.utils.gw_injection import (
    generate_simulated_waveform,
    inject_waveform
)

class TestBBHInj(unittest.TestCase):
    def setUp(self):
        approximant = "SEOBNRv4"
        m1, m2 = 57.0, 33.0
        delta_t, desired_length = 1./4096, 1.
        f_lower = 20
        snr = 7.5

        noise = np.load("gw_data/dataset/gwaves/L1-1175789870_1175805747_0.npy")
        bbh_waveform = generate_simulated_waveform(approximant, m1, m2, f_lower, delta_t)
        self.injected_signal, self.scaling_factor = inject_waveform(noise, bbh_waveform, snr, desired_length=desired_length)

    def test_inj_duration(self):
        assert(self.injected_signal.shape == (4096,))


class TestBNSInj(unittest.TestCase):
    def setUp(self):
        approximant = "TaylorF2"
        m1, m2 = 1.4, 1.2
        delta_t, desired_length = 1./4096, 1.
        f_lower = 20
        snr = 7.5

        noise = np.load("gw_data/dataset/gwaves/L1-1175789870_1175805747_0.npy")
        bns_waveform = generate_simulated_waveform(approximant, m1, m2, f_lower, delta_t)
        self.injected_signal, self.scaling_factor = inject_waveform(noise, bns_waveform, snr, desired_length=desired_length)

    def test_inj_duration(self):
        assert(self.injected_signal.shape == (4096,))
