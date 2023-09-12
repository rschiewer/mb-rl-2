import unittest

import torch
import numpy as np

from mdm.utils.torch_tools import *


class TorchUtilsTest(unittest.TestCase):

    def test_weighted_mean_bool_mask_0(self):
        x = torch.tensor([1, 2, 3])
        mask = torch.tensor([0, 0, 0])
        x_mean = masked_mean(x, mask)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), 2))

        mask = torch.tensor([1, 0, 0])
        x_mean = masked_mean(x, mask)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), 5 / 2))

        mask = torch.tensor([1, 1, 0])
        x_mean = masked_mean(x, mask)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), 3))

        mask = torch.tensor([1, 1, 1])
        x_mean = masked_mean(x, mask)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), 0))

        mask = torch.tensor([1, 0, 1])
        x_mean = masked_mean(x, mask)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), 2))

    def test_weighted_mean_bool_mask_1(self):
        x = torch.tensor([[1, 2, 3], [2, 3, 5]])
        mask = torch.tensor([[0, 0, 0], [0, 1, 0]])
        x_mean = masked_mean(x, mask)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), 13 / 5))

        mask = torch.tensor([[0, 0, 0], [0, 1, 0]])
        x_mean = masked_mean(x, mask, 0)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), [3 / 2, 2, 4]).all())

        mask = torch.tensor([[0, 0, 0], [0, 1, 0]])
        x_mean = masked_mean(x, mask, 1)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), [2, 7 / 2]).all())

    def test_weighted_mean_float_mask_0(self):
        x = torch.tensor([1, 2, 3])
        mask = torch.tensor([0.5, 0, 0])
        x_mean = masked_mean(x, mask)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), 5.5 / 2.5))

        mask = torch.tensor([0, 0.8, 0.9])
        x_mean = masked_mean(x, mask)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), (1 + 0.2 * 2 + 0.1 * 3) / 1.3))

    def test_weighted_mean_float_mask_1(self):
        x = torch.tensor([[1, 2, 3], [2, 3, 5]])
        mask = torch.tensor([[0, 0.8, 0.9], [1, 1, 1]])
        x_mean = masked_mean(x, mask)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), (1 + 0.2 * 2 + 0.1 * 3) / 1.3))

        mask = torch.tensor([[0, 0.8, 0.9], [1, 1, 1]])
        x_mean = masked_mean(x, mask)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), (1 + 0.2 * 2 + 0.1 * 3) / 1.3))

    def test_weighted_mean_tensor_mask_shape_mismatch(self):
        x = torch.tensor([[[1, 2, 3], [2, 3, 5]],
                          [[1, 2, 3], [2, 3, 5]]])
        mask = torch.tensor([[0, 1], [1, 0]])
        x_mean = masked_mean(x, mask)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), 16 / 6))

    def test_weighted_mean_keep_dim(self):
        x = torch.tensor([[1, 2, 3], [2, 3, 5]])
        mask = torch.tensor([[0, 0.8, 0.9], [1, 1, 1]])
        x_mean = masked_mean(x, mask, keepdim=True)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), (1 + 0.2 * 2 + 0.1 * 3) / 1.3))
        self.assertEqual(x_mean.shape, (1, 1))

        mask = torch.tensor([[0, 0, 0], [0, 1, 0]])
        x_mean = masked_mean(x, mask, 0, keepdim=True)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), [[3 / 2, 2, 4]]).all())
        self.assertEqual(x_mean.shape, (1, 3))

        mask = torch.tensor([[0, 0, 0], [0, 1, 0]])
        x_mean = masked_mean(x, mask, 1, keepdim=True)
        self.assertTrue(np.isclose(x_mean.detach().cpu().numpy(), [[2], [7 / 2]]).all())
        self.assertEqual(x_mean.shape, (2, 1))

    def test_weighted_var_bool_mask_0(self):
        x = torch.tensor([1, 2, 3], dtype=torch.float32)
        mask = torch.tensor([0, 0, 0])
        x_var = masked_var(x, mask)
        self.assertTrue(np.isclose(x_var.detach().cpu().numpy(), x.var(unbiased=False).detach().cpu().numpy()))

        mask = torch.tensor([1, 0, 0])
        x_var = masked_var(x, mask)
        self.assertTrue(np.isclose(x_var.detach().cpu().numpy(), ((2 - 2.5) ** 2 + (3 - 2.5) ** 2) / 2))

        mask = torch.tensor([1, 1, 0])
        x_var = masked_var(x, mask)
        self.assertTrue(np.isclose(x_var.detach().cpu().numpy(), 0))

        mask = torch.tensor([1, 1, 1])
        x_var = masked_var(x, mask)
        self.assertTrue(np.isclose(x_var.detach().cpu().numpy(), 0))

        mask = torch.tensor([0, 1, 0])
        x_var = masked_var(x, mask)
        self.assertTrue(np.isclose(x_var.detach().cpu().numpy(), ((1 - 2) ** 2 + (3 - 2) ** 2) / 2))

    def test_weighted_var_bool_mask_1(self):
        x = torch.tensor([[1, 2, 3], [2, 3, 5]], dtype=torch.float32)
        mask = torch.tensor([[0, 0, 0], [0, 1, 0]])
        x_var = masked_var(x, mask)
        self.assertTrue(np.isclose(x_var.detach().cpu().numpy(), np.var([1, 2, 3, 2, 5])))

        mask = torch.tensor([[0, 0, 0], [0, 1, 0]])
        x_var = masked_var(x, mask, 0)
        self.assertTrue(np.isclose(x_var.detach().cpu().numpy(), [0.25, 0, 1]).all())

        mask = torch.tensor([[0, 0, 0], [0, 1, 0]])
        x_var = masked_var(x, mask, 1)
        self.assertTrue(np.isclose(x_var.detach().cpu().numpy(), [2 / 3, 4.5 / 2]).all())

    def test_weighted_var_tensor_mask_shape_mismatch(self):
        x = torch.tensor([[[1, 2, 3], [2, 3, 5]],
                          [[1, 2, 3], [2, 3, 5]]])
        mask = torch.tensor([[0, 1], [1, 0]])
        x_var = masked_var(x, mask)
        self.assertTrue(np.isclose(x_var.detach().cpu().numpy(), np.var([1, 2, 3, 2, 3, 5])))

    def test_running_mean_std(self):
        d_batch = 32
        running_mean_std = RunningMeanStd(shape=(2, 1))
        v0 = torch.ones(1, 2, 1)
        v1 = torch.full((1, 2, 1), fill_value=2)

        self.assertTrue((running_mean_std.mean == 0).all())
        self.assertTrue((running_mean_std.var == 1).all())

        running_mean_std.update(v0)


if __name__ == '__main__':
    unittest.main()
