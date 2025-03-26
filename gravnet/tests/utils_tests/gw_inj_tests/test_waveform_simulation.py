import unittest
import numpy as np
from gravnet.utils.gw_injection import (
    generate_simulated_waveform,
    adjust_simulated_waveform
)

class TestBBHSim(unittest.TestCase):
    def setUp(self):
        approximant = "SEOBNRv4"
        m1, m2 = 57.0, 33.0
        f_lower, delta_t, desired_length = 20, 1./4096, 1.
        self.bbh_waveform = generate_simulated_waveform(
            approximant, m1, m2, f_lower, delta_t
        )
        self.bbh_waveform = adjust_simulated_waveform(self.bbh_waveform, desired_length)

    def test_bbh_duration(self):
        assert(self.bbh_waveform.data.shape == (4096,))

    def test_bbh_peak_idx(self):
        peak_idx = np.argmax(np.abs(self.bbh_waveform.data))
        assert(0.925 <= peak_idx/4096 <= 0.975)

class TestBNSSim(unittest.TestCase):
    def setUp(self):
        approximant = "TaylorF2"
        m1, m2 = 1.4, 1.1
        f_lower, delta_t, desired_length = 20, 1./4096, 1.
        self.bns_waveform = generate_simulated_waveform(
            approximant, m1, m2, f_lower, delta_t
        )
        self.bns_waveform = adjust_simulated_waveform(self.bns_waveform, desired_length)

    def test_bbh_duration(self):
        assert(self.bns_waveform.data.shape == (4096,))

    def test_bbh_peak_idx(self):
        peak_idx = np.argmax(np.abs(self.bns_waveform.data))
        assert(0.925 <= peak_idx/4096 <= 0.975)
