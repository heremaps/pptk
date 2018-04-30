import unittest
import numpy as np
from pptk.vfuncs import vfuncs


class TestAdd(unittest.TestCase):

    def setUp(self):
        self.A = np.float32(np.arange(9).reshape((3, 3)))
        self.a = self.A[0]
        self.b = np.float32(np.array([[1], [2], [3]]))
        self.v = np.hstack((self.b, self.b))
        self.w = np.float32(np.ndarray((0, 3)))

    def test_single_scalar(self):
        A = self.A
        a = self.a
        b = self.b
        v = self.v
        w = self.w

        Y = [A, A, A]
        z_gt = A + 1

        # add integer
        X = [1]
        Z = vfuncs._add(X, Y)
        for z in Z:
            self.assertTrue(z.tolist() == z_gt.tolist())
        Z = vfuncs._add(Y, X)
        for z in Z:
            self.assertTrue(z.tolist() == z_gt.tolist())

        # add float
        X = [1.0]
        Z = vfuncs._add(X, Y)
        for z in Z:
            self.assertTrue(z.tolist() == z_gt.tolist())
        Z = vfuncs._add(Y, X)
        for z in Z:
            self.assertTrue(z.tolist() == z_gt.tolist())

        # add boolean
        X = [True]
        Z = vfuncs._add(X, Y)
        for z in Z:
            self.assertTrue(z.tolist() == z_gt.tolist())
        Z = vfuncs._add(Y, X)
        for z in Z:
            self.assertTrue(z.tolist() == z_gt.tolist())

        # numpy.float32
        M = np.float32(np.array([1.0]))
        X = [M[0]]
        Z = vfuncs._add(X, Y)
        for z in Z:
            self.assertTrue(z.tolist() == z_gt.tolist())
        Z = vfuncs._add(Y, X)
        for z in Z:
            self.assertTrue(z.tolist() == z_gt.tolist())

    def test_input_checking(self):
        A = self.A
        a = self.a
        b = self.b
        v = self.v
        w = self.w

        X = [A, A]
        Y = [A, A, A]
        with self.assertRaises(ValueError):
            vfuncs._add(X, Y)
        with self.assertRaises(TypeError):
            vfuncs._add(1, 2, 3)
        with self.assertRaises(TypeError):
            vfuncs._add(1, X)
        with self.assertRaises(ValueError):
            vfuncs._add([], X)

    def test_broadcasting(self):
        A = self.A
        a = self.a
        b = self.b
        v = self.v
        w = self.w

        # broadcast rows
        X = [A, A, A]
        Y = [a, a, a]
        Z = vfuncs._add(X, Y)
        for z in Z:
            self.assertTrue(z.tolist() == (A + a).tolist())
        Z = vfuncs._add(Y, X)
        for z in Z:
            self.assertTrue(z.tolist() == (a + A).tolist())

        # broadcast columns
        X = [A, A, A]
        Y = [b, b, b]
        Z = vfuncs._add(X, Y)
        for z in Z:
            self.assertTrue(z.tolist() == (A + b).tolist())
        Z = vfuncs._add(Y, X)
        for z in Z:
            self.assertTrue(z.tolist() == (b + A).tolist())

        # outer sum
        X = [a, a, a]
        Y = [b, b, b]
        z_gt = a + b
        Z = vfuncs._add(X, Y)
        for z in Z:
            self.assertTrue(z.tolist() == z_gt.tolist())

        # shapes incompatible for broadcasting
        X = [A, A, A]
        Y = [v, v, v]
        with self.assertRaises(ValueError):
            vfuncs._add(X, Y)
        with self.assertRaises(ValueError):
            vfuncs._add(Y, X)

        # adding zero-row matrix to one-row matrix succeeds
        X = [a, a, a]
        Y = [w, w, w]
        Z = vfuncs._add(X, Y)
        for z in Z:
            self.assertTrue(z.tolist() == [] and z.shape == (0, 3))
        Z = vfuncs._add(Y, X)
        for z in Z:
            self.assertTrue(z.tolist() == [] and z.shape == (0, 3))

        # adding zero-row matrix to anything else should fail
        X = [A, A, A]
        Y = [w, w, w]
        with self.assertRaises(ValueError):
            vfuncs._add(X, Y)
        with self.assertRaises(ValueError):
            vfuncs._add(Y, X)

    def test_array_checking(self):
        A = self.A
        a = self.a
        b = self.b
        v = self.v
        w = self.w
        V = np.float32(np.random.rand(3, 3, 3))

        # bad array occurs as single item in list
        X = [V]
        Y = [a, a, a]
        with self.assertRaises(ValueError):
            vfuncs._add(X, Y)
        with self.assertRaises(ValueError):
            vfuncs._add(Y, X)

        # bad array occurs at second position in list
        X = [A, V, A]
        Y = [a, a, a]
        with self.assertRaises(ValueError):
            vfuncs._add(X, Y)
        with self.assertRaises(ValueError):
            vfuncs._add(Y, X)


class TestIndexing(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.A = np.matrix(np.float32(np.random.rand(10, 11)))
        self.J = [2, 5]
        self.S = slice(None, None, None)
        self.B = np.array([
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            True,
            False,
            False])

    def test_check_inputs(self):
        A = self.A
        J = self.J
        S = self.S
        X = [A, A, A]

        # simultaneous indexing of rows and columns disallowed
        with self.assertRaises(TypeError):
            vfuncs._idx(X, ([J], [J]))

        # indexing just rows should not give error
        Y = vfuncs._idx(X, ([J], S))
        for y in Y:
            self.assertTrue(y.tolist() == A[J].tolist())

        # indexing just columns should not give error
        Y = vfuncs._idx(X, (S, [J]))
        for y in Y:
            self.assertTrue(y.tolist() == A[:, J].tolist())

        # indexing X=[A, A, A] with [J, J] should give error
        with self.assertRaises(ValueError):
            vfuncs._idx(X, ([J, J], S))

        # index [A] with [J]
        Y = vfuncs._idx([A], ([J], S))
        for y in Y:
            self.assertTrue(y.tolist() == A[J, :].tolist())
        Y = vfuncs._idx([A], (S, [J]))
        for y in Y:
            self.assertTrue(y.tolist() == A[:, J].tolist())

        # index [A] with [J, J, J]
        Y = vfuncs._idx([A], ([J, J, J], S))
        for y in Y:
            self.assertTrue(y.tolist() == A[J, :].tolist())
        Y = vfuncs._idx([A], (S, [J, J, J]))
        for y in Y:
            self.assertTrue(y.tolist() == A[:, J].tolist())

        # index [A, A, A] with [J]
        Y = vfuncs._idx([A, A, A], ([J], S))
        for y in Y:
            self.assertTrue(y.tolist() == A[J, :].tolist())
        Y = vfuncs._idx([A, A, A], (S, [J]))
        for y in Y:
            self.assertTrue(y.tolist() == A[:, J].tolist())

        # index [A, A, A] with [J, J, J]
        Y = vfuncs._idx([A, A, A], ([J, J, J], S))
        for y in Y:
            self.assertTrue(y.tolist() == A[J, :].tolist())
        Y = vfuncs._idx([A, A, A], (S, [J, J, J]))
        for y in Y:
            self.assertTrue(y.tolist() == A[:, J].tolist())

        # indexing [] with [J] should give error
        with self.assertRaises(ValueError):
            vfuncs._idx([], ([J], S))
        with self.assertRaises(ValueError):
            vfuncs._idx([], (S, [J]))

        # indexing [] with [J, J] should give error
        with self.assertRaises(ValueError):
            vfuncs._idx([], ([J, J], S))
        with self.assertRaises(ValueError):
            vfuncs._idx([], (S, [J, J]))

        # indexing [A] with [] should give error
        with self.assertRaises(ValueError):
            vfuncs._idx([A], ([], S))
        with self.assertRaises(ValueError):
            vfuncs._idx([A], (S, []))

        # indexing [A, A] with [] should give error
        with self.assertRaises(ValueError):
            vfuncs._idx([A, A], ([], S))
        with self.assertRaises(ValueError):
            vfuncs._idx([A, A], (S, []))

    def test_slicing(self):
        A = self.A
        J = self.J
        S = self.S
        X = [A, A, A]

        # out of bound slicing results in empty array
        S1 = slice(12, None, None)
        Y = vfuncs._idx(X, (S1, S))
        for y in Y:
            self.assertTrue(
                y.shape == A[S1].shape and
                y.tolist() == A[S1].tolist())
        Y = vfuncs._idx(X, (S, S1))
        for y in Y:
            self.assertTrue(
                y.shape == A[:, S1].shape and
                y.tolist() == A[:, S1].tolist())

        # slices with negative indices
        S2 = slice(-2, None, None)
        Y = vfuncs._idx(X, (S2, S))
        for y in Y:
            self.assertTrue(y.tolist() == A[S2].tolist())
        Y = vfuncs._idx(X, (S, S2))
        for y in Y:
            self.assertTrue(y.tolist() == A[:, S2].tolist())

        # simultaneous row and column slicing
        Y = vfuncs._idx(X, (slice(None, None, 2), slice(None, None, 3)))
        for y in Y:
            self.assertTrue(y.tolist() == A[::2, ::3].tolist())

    def test_scalar(self):
        # indexing with scalar
        A = self.A
        X = [A, A, A]
        S = slice(None, None, None)

        # index rows (single scalar index for all list items)
        Y = vfuncs._idx(X, ([0], S))
        for y in Y:
            self.assertTrue(y.tolist() == A[0].tolist())

        # index rows (separate scalar index per list items)
        Y = vfuncs._idx(X, ([0, 0, 0], S))
        for y in Y:
            self.assertTrue(y.tolist() == A[0].tolist())

        # index columns (single scalar index for all list items)
        Y = vfuncs._idx(X, (S, [0]))
        for y in Y:
            self.assertTrue(y.tolist() == A[:, 0].tolist())

        # index columns (separate scalar index per list items)
        Y = vfuncs._idx(X, (S, [0, 0, 0]))
        for y in Y:
            self.assertTrue(y.tolist() == A[:, 0].tolist())

    def test_pylist(self):
        # indexing with python list of indices
        A = self.A
        S = self.S
        X = [A, A, A]

        # indexing with list containing non-int()-able item
        # should raise exception
        class Foo:
            pass
        J = [1, Foo(), 2]
        with self.assertRaises(TypeError):
            vfuncs._idx(X, ([J], S))

    def test_boolean(self):
        # indexing with array of booleans
        A = self.A
        S = self.S
        B = self.B
        B2 = np.r_[B, False]
        X = [A, A, A]

        Y = vfuncs._idx(X, ([B], S))
        for y in Y:
            self.assertTrue(y.tolist() == A[B].tolist())
        Y = vfuncs._idx(X, (S, [B2]))
        for y in Y:
            self.assertTrue(y.tolist() == A[:, B2].tolist())
        Y = vfuncs._idx(X, ([B, B, B], S))
        for y in Y:
            self.assertTrue(y.tolist() == A[B, :].tolist())
        Y = vfuncs._idx(X, (S, [B2, B2, B2]))
        for y in Y:
            self.assertTrue(y.tolist() == A[:, B2].tolist())

        # length of boolean array allowed to exceed that of "indexee"
        # as long as exceeding entries are all False
        Y = vfuncs._idx(X, ([B[:3]], S))
        for y in Y:
            self.assertTrue(y.tolist() == A[:3, :][B[:3], :].tolist())
        B2 = np.hstack((B, np.array([False, False, False])))
        Y = vfuncs._idx(X, ([B2], S))
        for y in Y:
            self.assertTrue(y.tolist() == A[B, :].tolist())

    def test_intarray(self):
        A = self.A
        S = self.S
        X = [A, A, A]

        # indexing with array of integers

        # negative indices

        # out of bound indices should raise exception
        J = np.array([42])
        with self.assertRaises(IndexError):
            vfuncs._idx(X, ([J], S))
        with self.assertRaises(IndexError):
            vfuncs._idx(X, (S, [J]))

        J = np.array([-13])
        with self.assertRaises(IndexError):
            vfuncs._idx(X, ([J], S))
        with self.assertRaises(IndexError):
            vfuncs._idx(X, (S, [J]))

    def test_mixed(self):
        A = self.A
        S = self.S
        J = [
            self.B,
            [1, 4, 5],
            [True, 2.0, False, 6],
            np.array([2, 3, 9], dtype=np.int32),
            np.array([-3, 9])
        ]
        J_1 = [
            self.B,
            [1, 4, 5],
            [1, 2, 0, 6],
            np.array([2, 3, 9], dtype=np.int32),
            np.array([-3, 9])
        ]
        J_2 = [
            np.r_[self.B, False],
            [1, 4, 5],
            [1, 2, 0, 6],
            np.array([2, 3, 9], dtype=np.int32),
            np.array([-3, 9])
        ]
        X = [A, A, A, A, A]
        Y = vfuncs._idx(X, (J, S))
        for i, y in enumerate(Y):
            self.assertTrue(y.tolist() == A[J_1[i]].tolist())
        Y = vfuncs._idx(X, (S, J))
        for i, y in enumerate(Y):
            self.assertTrue(y.tolist() == A[:, J_2[i]].tolist())


class TestDot(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.A1 = np.float32(np.random.rand(2, 5))
        self.A2 = np.float32(np.random.rand(1, 5))
        self.B1 = np.float32(np.random.rand(5, 3))
        self.B2 = np.float32(np.random.rand(5, 2))

    def test_simple(self):
        A1 = self.A1
        A2 = self.A2
        B1 = self.B1
        B2 = self.B2

        # [A1]*[B1]
        Y = vfuncs._dot([A1], [B1])
        self.assertTrue(len(Y) == 1)
        self.assertTrue(np.allclose(Y[0], np.dot(A1, B1)))

        # [A1]*[B1,B2]
        Y = vfuncs._dot([A1], [B1, B2])
        self.assertTrue(len(Y) == 2)
        self.assertTrue(np.allclose(Y[0], np.dot(A1, B1)))
        self.assertTrue(np.allclose(Y[1], np.dot(A1, B2)))

        # [A1,A2]*[B1]
        Y = vfuncs._dot([A1, A2], [B1])
        self.assertTrue(len(Y) == 2)
        self.assertTrue(np.allclose(Y[0], np.dot(A1, B1)))
        self.assertTrue(np.allclose(Y[1], np.dot(A2, B1)))

        # [A1,A2]*[B1,B2]
        Y = vfuncs._dot([A1, A2], [B1, B2])
        self.assertTrue(len(Y) == 2)
        self.assertTrue(np.allclose(Y[0], np.dot(A1, B1)))
        self.assertTrue(np.allclose(Y[1], np.dot(A2, B2)))

        # incompatible shapes should raise value error
        with self.assertRaises(ValueError):
            vfuncs._dot([A1], [A2])

    def test_multiply_by_scalar(self):
        A = self.A1

        # [A,A,A]*[.5]
        Y = vfuncs._dot([A, A, A], [.5])
        for y in Y:
            self.assertTrue(y.tolist() == (A*.5).tolist())

        # [.2]*[.5]
        Y = vfuncs._dot([.2], [.5])
        self.assertTrue(Y[0].tolist() == [[np.float32(.2) * np.float32(.5)]])


class TestTranspose(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.A = np.float32(np.random.rand(3, 3))
        self.B = np.float32(np.random.rand(3, 4))

    def test_simple(self):
        A = self.A
        B = self.B

        Y = vfuncs._transpose([A, B])
        self.assertTrue(Y[0].tolist() == A.T.tolist())
        self.assertTrue(Y[1].tolist() == B.T.tolist())


if __name__ == '__main__':
    unittest.main()
