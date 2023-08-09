import unittest

import torch
import numpy as np

from mdm.utils.torch_tools import bin_every_k_steps
from mdm.utils.utils import valid_subtrajectories_unbiased


class TorchUtilsTest(unittest.TestCase):

    def test_bin_every_k_steps(self):
        data = torch.tensor([
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 0, 0, 0, 1, 1, 1, 1, 2],
            [1, 0, 0, 0, 2, 1, 1, 1, 3],
            [1, 0, 0, 0, 2, 1, 1, 1, 4]
        ], dtype=torch.float32)
        data = data.unsqueeze(-1)  # add data dimension

        # bin with 3 steps and zero padding
        target_mean_3_steps_0 = np.array([
            [  0, 3/3, 6/3],
            [  0, 2/3, 4/3],
            [1/3, 3/3, 5/3],
            [1/3, 3/3, 6/3]
        ], dtype=np.float32)
        binned_3_steps = bin_every_k_steps(data, 3, 0)
        mean_3_steps = binned_3_steps.mean(dim=2).squeeze(-1)
        diff = np.sum(target_mean_3_steps_0 - mean_3_steps.detach().cpu().numpy())
        self.assertTrue(np.isclose(diff, 0))

        # bin with 4 steps and zero padding
        target_mean_4_steps_0 = np.array([
            [1/4, 6/4, 2/4],
            [  0, 4/4, 2/4],
            [1/4, 5/4, 3/4],
            [1/4, 5/4, 4/4]
        ], dtype=np.float32)
        binned_4_steps = bin_every_k_steps(data, 4, 0)
        mean_4_steps = binned_4_steps.mean(dim=2).squeeze(-1)
        diff = np.sum(target_mean_4_steps_0 - mean_4_steps.detach().cpu().numpy())
        self.assertTrue(np.isclose(diff, 0))

        # bin with 3 steps and ones padding
        target_mean_4_steps_1 = np.array([
            [1/4, 6/4, 5/4],
            [  0, 4/4, 5/4],
            [1/4, 5/4, 6/4],
            [1/4, 5/4, 7/4]
        ], dtype=np.float32)
        binned_4_steps = bin_every_k_steps(data, 4, 1)
        mean_4_steps = binned_4_steps.mean(dim=2).squeeze(-1)
        diff = np.sum(target_mean_4_steps_1 - mean_4_steps.detach().cpu().numpy())
        self.assertTrue(np.isclose(diff, 0))


if __name__ == '__main__':
    unittest.main()