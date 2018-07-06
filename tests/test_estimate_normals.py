import unittest
import pptk
import numpy as np


class TestEstimateNormals(unittest.TestCase):
    def test_rand100(self):
        np.random.seed(0)
        x = pptk.rand(100, 3)

        pptk.estimate_normals(x, 5, 0.2,
                              verbose=False)
        pptk.estimate_normals(x, 5, 0.2,
                              output_eigenvalues=True,
                              verbose=False)
        pptk.estimate_normals(x, 5, 0.2,
                              output_all_eigenvectors=True,
                              verbose=False)
        pptk.estimate_normals(x, 5, 0.2,
                              output_eigenvalues=True,
                              output_all_eigenvectors=True,
                              verbose=False)

        x = np.float32(x)
        pptk.estimate_normals(x, 5, 0.2,
                              verbose=False)
        pptk.estimate_normals(x, 5, 0.2,
                              output_eigenvalues=True,
                              verbose=False)
        pptk.estimate_normals(x, 5, 0.2,
                              output_all_eigenvectors=True,
                              verbose=False)
        pptk.estimate_normals(x, 5, 0.2,
                              output_eigenvalues=True,
                              output_all_eigenvectors=True,
                              verbose=False)

    def test_output_types(self):
        # test all 8 combinations of output_* switches

        y = np.zeros((10, 3), dtype=np.float64)
        y[:, 0] = np.arange(10)
        k = 4
        r = 3
        evals = np.array([2. / 3] + [5. / 4] * 8 + [2. / 3])
        evecs = np.array([[1., 0., 0.]] * 10)
        nbhd_sizes = np.array([3] + [4] * 8 + [3])

        # combo 1: (0, 0, 0)
        z = pptk.estimate_normals(y, k, r,
                                  output_eigenvalues=False,
                                  output_all_eigenvectors=False,
                                  output_neighborhood_sizes=False,
                                  verbose=False)
        self.assertTrue(isinstance(z, np.ndarray) and z.shape == (10, 3))

        # combo 2: (0, 0, 1)
        z = pptk.estimate_normals(y, k, r,
                                  output_eigenvalues=False,
                                  output_all_eigenvectors=False,
                                  output_neighborhood_sizes=True,
                                  verbose=False)
        self.assertTrue(
            isinstance(z, tuple) and len(z) == 3 and
            [type(i) for i in z] == [np.ndarray, type(None), np.ndarray] and
            z[0].shape == (10, 3) and z[2].shape == (10, ))
        self.assertTrue(np.all(z[2] == nbhd_sizes))

        # combo 3: (0, 1, 0)
        z = pptk.estimate_normals(y, k, r,
                                  output_eigenvalues=False,
                                  output_all_eigenvectors=True,
                                  output_neighborhood_sizes=False,
                                  verbose=False)
        self.assertTrue(isinstance(z, np.ndarray) and z.shape == (10, 3, 3))

        # combo 4: (0, 1, 1)
        z = pptk.estimate_normals(y, k, r,
                                  output_eigenvalues=False,
                                  output_all_eigenvectors=True,
                                  output_neighborhood_sizes=True,
                                  verbose=False)
        self.assertTrue(
            isinstance(z, tuple) and len(z) == 3 and
            [type(i) for i in z] == [np.ndarray, type(None), np.ndarray] and
            z[0].shape == (10, 3, 3) and z[2].shape == (10, ))
        self.assertTrue(np.allclose(z[0][:, -1, :], evecs))
        self.assertTrue(np.all(z[2] == nbhd_sizes))

        # combo 5: (1, 0, 0)
        z = pptk.estimate_normals(y, k, r,
                                  output_eigenvalues=True,
                                  output_all_eigenvectors=False,
                                  output_neighborhood_sizes=False,
                                  verbose=False)
        self.assertTrue(
            isinstance(z, tuple) and len(z) == 3 and
            [type(i) for i in z] == [np.ndarray, np.ndarray, type(None)] and
            z[0].shape == (10, 3) and z[1].shape == (10, ))

        # combo 6: (1, 0, 1)
        z = pptk.estimate_normals(y, k, r,
                                  output_eigenvalues=True,
                                  output_all_eigenvectors=False,
                                  output_neighborhood_sizes=True,
                                  verbose=False)
        self.assertTrue(
            isinstance(z, tuple) and len(z) == 3 and
            [type(i) for i in z] == [np.ndarray, np.ndarray, np.ndarray] and
            [i.shape for i in z] == [(10, 3), (10, ), (10, )])
        self.assertTrue(np.all(z[2] == nbhd_sizes))

        # combo 7: (1, 1, 0)
        z = pptk.estimate_normals(y, k, r,
                                  output_eigenvalues=True,
                                  output_all_eigenvectors=True,
                                  output_neighborhood_sizes=False,
                                  verbose=False)
        self.assertTrue(
            isinstance(z, tuple) and len(z) == 3 and
            [type(i) for i in z] == [np.ndarray, np.ndarray, type(None)] and
            z[0].shape == (10, 3, 3) and z[1].shape == (10, 3))
        self.assertTrue(np.allclose(z[0][:, -1, :], evecs))
        self.assertTrue(np.allclose(z[1][:, -1], evals))

        # combo 8: (1, 1, 1)
        z = pptk.estimate_normals(y, k, r,
                                  output_eigenvalues=True,
                                  output_all_eigenvectors=True,
                                  output_neighborhood_sizes=True,
                                  verbose=False)
        self.assertTrue(
            isinstance(z, tuple) and len(z) == 3 and
            [type(i) for i in z] == [np.ndarray, np.ndarray, np.ndarray] and
            [i.shape for i in z] == [(10, 3, 3), (10, 3), (10, )])
        self.assertTrue(np.allclose(z[0][:, -1, :], evecs))
        self.assertTrue(np.allclose(z[1][:, -1], evals))
        self.assertTrue(np.all(z[2] == nbhd_sizes))

    def test_neighborhood_types(self):
        pass

    def test_subsample(self):
        y = np.zeros((10, 3), dtype=np.float64)
        y[:, 0] = np.arange(10)
        k = 4
        r = 3
        evals = np.array([2. / 3] + [5. / 4] * 8 + [2. / 3])
        evecs = np.array([[1., 0., 0.]] * 10)
        nbhd_sizes = np.array([3] + [4] * 8 + [3])

        s = [0, 3, 4, -1]

        mask = np.zeros(10, dtype=np.bool)
        mask[s] = True
        z = pptk.estimate_normals(y, k, r,
                                  subsample=mask,
                                  output_eigenvalues=True,
                                  output_all_eigenvectors=True,
                                  output_neighborhood_sizes=True,
                                  verbose=False)
        self.assertTrue(np.allclose(z[0][:, -1, :], evecs[s]))
        self.assertTrue(np.allclose(z[1][:, -1], evals[s]))
        self.assertTrue(np.all(z[2] == nbhd_sizes[s]))

        mask = np.zeros(2, dtype=np.bool)
        with self.assertRaises(ValueError):
            pptk.estimate_normals(y, k, r,
                                  subsample=mask,
                                  output_eigenvalues=True,
                                  output_all_eigenvectors=True,
                                  output_neighborhood_sizes=True,
                                  verbose=False)

        z = pptk.estimate_normals(y, k, r,
                                  subsample=s,
                                  output_eigenvalues=True,
                                  output_all_eigenvectors=True,
                                  output_neighborhood_sizes=True,
                                  verbose=False)
        self.assertTrue(np.allclose(z[0][:, -1, :], evecs[s]))
        self.assertTrue(np.allclose(z[1][:, -1], evals[s]))
        self.assertTrue(np.all(z[2] == nbhd_sizes[s]))

if __name__ == '__main__':
    unittest.main()
