import unittest
import pypcl
import numpy as np


class TestEstimateNormals(unittest.TestCase):
    def test_rand100(self):
        np.random.seed(0)
        x = pypcl.rand(100,3)
        pypcl.estimate_normals(x, 5, 0.2)
        pypcl.estimate_normals(x, 5, 0.2, output_eigenvalues=True)
        pypcl.estimate_normals(x, 5, 0.2, output_all_eigenvectors=True)
        pypcl.estimate_normals(x, 5, 0.2, output_eigenvalues=True,
                               output_all_eigenvectors=True)

        x = np.float32(x)
        pypcl.estimate_normals(x, 5, 0.2)
        pypcl.estimate_normals(x, 5, 0.2, output_eigenvalues=True)
        pypcl.estimate_normals(x, 5, 0.2, output_all_eigenvectors=True)
        pypcl.estimate_normals(x, 5, 0.2, output_eigenvalues=True,
                               output_all_eigenvectors=True)


if __name__ == '__main__':
    unittest.main()
