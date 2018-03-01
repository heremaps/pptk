import unittest
import pypcl
import numpy as np


class TestCentroids(unittest.TestCase):
    def test_centroids(self):
        np.random.seed(0)
        x = np.float32(pypcl.rand(100, 3))
        n = x.NBHDS(k=3)
        y = np.vstack(pypcl.MEAN(x[n], axis=0).evaluate())
        self.assertTrue(y.shape == (100, 3))


if __name__ == '__main__':
    unittest.main()
