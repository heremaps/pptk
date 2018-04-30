import unittest
import pptk
import numpy as np


class TestEstimateNormals(unittest.TestCase):
    def test_rand100(self):
        np.random.seed(0)
        x = pptk.rand(100,3)
        pptk.estimate_normals(x, 5, 0.2)
        pptk.estimate_normals(x, 5, 0.2, output_eigenvalues=True)
        pptk.estimate_normals(x, 5, 0.2, output_all_eigenvectors=True)
        pptk.estimate_normals(x, 5, 0.2, output_eigenvalues=True,
                               output_all_eigenvectors=True)

        x = np.float32(x)
        pptk.estimate_normals(x, 5, 0.2)
        pptk.estimate_normals(x, 5, 0.2, output_eigenvalues=True)
        pptk.estimate_normals(x, 5, 0.2, output_all_eigenvectors=True)
        pptk.estimate_normals(x, 5, 0.2, output_eigenvalues=True,
                               output_all_eigenvectors=True)


if __name__ == '__main__':
    unittest.main()
