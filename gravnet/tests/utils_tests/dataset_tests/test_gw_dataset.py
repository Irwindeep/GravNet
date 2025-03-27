import unittest
from gravnet.utils.datasets import GWDataset

class TestGWDataset(unittest.TestCase):
    def test_data_lengths(self):
        train_dataset = GWDataset(root="gw_data/", split="train")
        test_dataset = GWDataset(root="gw_data/", split="test")
        val_dataset = GWDataset(root="gw_data/", split="val")

        assert(len(train_dataset) == 102550)
        assert(len(test_dataset) == 128190)
        assert(len(val_dataset) == 25639)
