import unittest
from utils.ray import *


class TestRayMethods(unittest.TestCase):
    def test_volumn_render(self):
        torch.manual_seed(42)
        sigmas = torch.rand((10, 64))
        rgbs = torch.rand((10, 64, 3))
        step_size = (6.0 - 2.0) / 64
        rendered_colors = volumn_render(sigmas, rgbs, step_size)

        correct = torch.tensor(
            [
                [0.5006, 0.3728, 0.4728],
                [0.4322, 0.3559, 0.4134],
                [0.4027, 0.4394, 0.4610],
                [0.4514, 0.3829, 0.4196],
                [0.4002, 0.4599, 0.4103],
                [0.4471, 0.4044, 0.4069],
                [0.4285, 0.4072, 0.3777],
                [0.4152, 0.4190, 0.4361],
                [0.4051, 0.3651, 0.3969],
                [0.3253, 0.3587, 0.4215],
            ]
        )
        self.assertTrue(torch.allclose(rendered_colors, correct, rtol=1e-4, atol=1e-4))

if __name__ == "__main__":
    unittest.main()