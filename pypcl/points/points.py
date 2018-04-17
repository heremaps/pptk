import numpy
import time
import ctypes
from ..kdtree import kdtree
from . import expr

__all__ = [
    'Points',
    'points',
    'zeros',
    'ones',
    'empty',
    'zeros_like',
    'ones_like',
    'empty_like',
    'rand',
    'load'
]


class Final(type):
    # copied from https://stackoverflow.com/questions/16056574
    def __new__(cls, name, bases, classdict):
        for b in bases:
            if isinstance(b, Final):
                raise TypeError("type " + b.__name__ +
                                " is not an acceptable base type")
        return type.__new__(cls, name, bases, dict(classdict))


def _memaddr(obj):
    return obj.ctypes.get_data()


def points(object, dtype=None, copy=True):
    if not copy and isinstance(object, numpy.ndarray) \
            and (dtype is None or dtype == object.dtype):
        ret = object.view(Points)
    else:
        temp = numpy.array(object, dtype=dtype, copy=False)
        ret = empty_like(temp)
        ret[:] = temp
    return ret


def zeros(shape, dtype=float):
    ret = Points(shape=shape, dtype=dtype)
    ret[:] = 0
    return ret


def ones(shape, dtype=float):
    ret = Points(shape=shape, dtype=dtype)
    ret[:] = 1
    return ret


def empty(shape, dtype=float):
    return Points(shape=shape, dtype=dtype)


def zeros_like(a, dtype=None):
    return zeros(a.shape, dtype=(a.dtype if dtype is None else dtype))


def ones_like(a, dtype=None):
    return zeros(a.shape, dtype=(a.dtype if dtype is None else dtype))


def empty_like(a, dtype=None):
    return zeros(a.shape, dtype=(a.dtype if dtype is None else dtype))


def rand(*dims):
    ret = empty(shape=dims, dtype=float)
    ret[:] = numpy.random.rand(*dims)
    return ret


def load(file, **kwargs):
    # wrapper around numpy.load
    # TODO: this copies to numpy array, then to a Points object;
    #       find way to avoid this extra copy
    return points(numpy.load(file, **kwargs))


class Points (numpy.ndarray):

    _last_modified = dict()

    # make Points non subclass-able to simplify write control
    # TODO: are there any use cases for subclassing Points?
    __metaclass__ = Final

    def __new__(cls, *args, **kwargs):
        return super(Points, cls).__new__(cls, *args, **kwargs)

    def __array_finalize__(self, obj):
        self._last_updated = None
        self._tree = None

        if obj is not None and not isinstance(obj, Points):
            # arrived at here via view() of a non-Points object
            raise TypeError('Detected attempt at creating Points-type '
                            'view on non-Points object.')

        if obj is None and not self.flags.owndata:
            raise TypeError('Detected attempt at creating Points-type '
                            'view on buffer object via __new__(buffer=...)')

        if obj is None:
            # arrived at here via __new__
            self._memsize = self.size * self.dtype.itemsize
            self._memloc = _memaddr(self)
        elif _memaddr(self) < obj._memloc or \
                _memaddr(self) >= obj._memloc + obj._memsize:
            # arrived at here via copy()
            self._memsize = self.size * self.dtype.itemsize
            self._memloc = _memaddr(self)
        else:
            # arrived at here via slicing/indexing
            # or view() of a Points object
            self._memsize = obj._memsize
            self._memloc = obj._memloc

        # cannot set writeable flag to False here,
        # because copy() performs assignment after __array_finalize__

    def __init__(self, *args, **kwargs):
        self.flags.writeable = False

    def copy(self):
        x = super(Points, self).copy()
        x.flags.writeable = False
        return x

    def _record_modify_time(self):
        Points._last_modified[self._memloc] = time.time()

    def _update_kd_tree(self):
        # if there is no recorded last modify time for self._memloc,
        # then self has either not been modified yet since creation,
        # or _last_modified dictionary has been cleared. Either way,
        # the k-d tree needs updating; we set the last modify time to
        # the current time to trigger this.
        if Points._last_modified.get(self._memloc) is None:
            Points._last_modified[self._memloc] = time.time()

        # note: None < x, for any number x
        build_time = None
        if self._last_updated is None \
                or self._last_updated <= Points._last_modified[self._memloc]:
            # note: do not need to explicitly call __del__()
            # as it is automatically called when overwritten
            build_time = time.time()
            self._tree = kdtree._build(self)
            build_time = time.time() - build_time
            self._last_updated = time.time()  # record time *after* build
        return build_time

    def nbhds(self, queries=None, k=1, r=None, verbose=False):
        self._update_kd_tree()
        return kdtree._query(self._tree, queries=queries, k=k, dmax=r)

    def NBHDS(self, queries=None, k=1, r=None, verbose=False):
        return expr.nbhds_op(self, queries, k, r)

    def _guard(self, f):
        def f_guarded(*args, **kwargs):
            if self.base is not None:
                self.base.flags.writeable = True
            self.flags.writeable = True
            ret = None
            try:
                ret = f(*args, **kwargs)
            finally:
                self.flags.writeable = False
                if self.base is not None:
                    self.base.flags.writeable = False
            self._record_modify_time()  # record time *after* computation
            return ret
        return f_guarded

    # override methods that modify object content to
    # record timestamp, signalling need for k-d tree update

    # inplace arithmetic methods
    # e.g. +=, -=, *=, /=, //=, %=, **=, <<=, >>=, &=, ^=, |=
    def __iadd__(self, other):
        return self._guard(super(Points, self).__iadd__)(other)

    def __isub__(self, other):
        return self._guard(super(Points, self).__isub__)(other)

    def __imul__(self, other):
        return self._guard(super(Points, self).__imul__)(other)

    def __idiv__(self, other):
        return self._guard(super(Points, self).__idiv__)(other)

    def __itruediv__(self, other):
        return self._guard(super(Points, self).__itruediv__)(other)

    def __ifloordiv__(self, other):
        return self._guard(super(Points, self).__ifloordiv__)(other)

    def __imod__(self, other):
        return self._guard(super(Points, self).__imod__)(other)

    def __ipow__(self, other):
        return self._guard(super(Points, self).__ipow__)(other)

    def __ilshift__(self, other):
        return self._guard(super(Points, self).__ilshift__)(other)

    def __irshift__(self, other):
        return self._guard(super(Points, self).__irshift__)(other)

    def __iand__(self, other):
        return self._guard(super(Points, self).__iand__)(other)

    def __ixor__(self, other):
        return self._guard(super(Points, self).__ixor__)(other)

    def __ior__(self, other):
        return self._guard(super(Points, self).__ior__)(other)

    # indexing and slicing operator
    def __setslice__(self, i, j, sequence):
        return self._guard(super(Points, self).__setslice__)(i, j, sequence)

    def __delslice__(self, i, j):
        return self._guard(super(Points, self).__delslice__)(i, j)

    def __getslice__(self, i, j):
        return super(Points, self).__getslice__(i, j)

    def __setitem__(self, key, value):
        return self._guard(super(Points, self).__setitem__)(key, value)

    def __delitem__(self, key):
        return self._guard(super(Points, self).__delitem__)(key)

    def __getitem__(self, key):
        if isinstance(key, expr.expression):
            return expr.index_op(
                expr._make_expression(self), key, slice(None, None, None))
        elif (isinstance(key, tuple) or isinstance(key, list)) \
                and any([isinstance(x, expr.expression) for x in key]):
            # key is a sequence containing at least one expression object
            if len(key) == 2 and (
                    isinstance(key[0], expr.expression)
                    and isinstance(key[1], slice)
                    or isinstance(key[0], slice)
                    and isinstance(key[1], expr.expression)):
                return expr.index_op(
                    expr._make_expression(self), key[0], key[1])
            else:
                raise TypeError(
                    'unsupported combination of types in index tuple: %s'
                    % repr((type(x) for x in key)))
        else:
            return super(Points, self).__getitem__(key)
