from ..vfuncs import vfuncs
from ..kdtree import kdtree
import numpy as np
import copy
import math

__all__ = [
    'MEAN',
    'SUM',
    'PROD',
    'ALL',
    'ANY',
    'MIN',
    'MAX',
    'ARGMIN',
    'ARGMAX',
    'EIGH',
    'DOT',
    'TRANSPOSE',
]

_num_chunks = 128
_min_chunk_size = 1000
_max_chunk_size = 5000


def MEAN(operand, axis=None):
    return unary_op(operand, vfuncs._mean, {'axis': axis})


def SUM(operand, axis=None):
    return unary_op(operand, vfuncs._sum, {'axis': axis})


def PROD(operand, axis=None):
    return unary_op(operand, vfuncs._prod, {'axis': axis})


def ALL(operand, axis=None):
    return unary_op(operand, vfuncs._all, {'axis': axis})


def ANY(operand, axis=None):
    return unary_op(operand, vfuncs._any, {'axis': axis})


def MIN(operand, axis=None):
    return unary_op(operand, vfuncs._min, {'axis': axis})


def MAX(operand, axis=None):
    return unary_op(operand, vfuncs._max, {'axis': axis})


def ARGMIN(operand, axis=None):
    return unary_op(operand, vfuncs._argmin, {'axis': axis})


def ARGMAX(operand, axis=None):
    return unary_op(operand, vfuncs._argmax, {'axis': axis})


def EIGH(operand):
    return unary_op(operand, vfuncs._eigh)


def DOT(left, right):
    return binary_op(left, right, vfuncs._dot)


def TRANSPOSE(operand):
    return unary_op(operand, vfuncs._transpose)


class expression(object):
    # interface enforcing methods
    def __init__(self):
        raise RuntimeError('expression class not meant for instantiation')

    def __len__(self):
        raise RuntimeError('attempting to take len()' +
                           'of non-instantiable expression object')

    @staticmethod
    def _check_operands():
        raise RuntimeError('method meant for overriding in subclass')

    def _evaluate_chunk(self, index, size):
        raise RuntimeError('method meant for overriding in subclass')

    # arithmetic operators
    def __add__(self, other):
        return binary_op(self, other, vfuncs._add)

    def __sub__(self, other):
        return binary_op(self, other, vfuncs._sub)

    def __mul__(self, other):
        return binary_op(self, other, vfuncs._mul)

    def __div__(self, other):
        return binary_op(self, other, vfuncs._div)

    # reflected arithmetic operators
    def __radd__(self, other):
        return binary_op(other, self, vfuncs._add)

    def __rsub__(self, other):
        return binary_op(other, self, vfuncs._sub)

    def __rmul__(self, other):
        return binary_op(other, self, vfuncs._mul)

    def __rdiv__(self, other):
        return binary_op(other, self, vfuncs._div)

    # other magic methods
    def __repr__(self):
        num_items = len(self)
        if num_items > 10:
            # evaluate first and last 3 items in expression
            # todo: handle stateful expressions
            results_beg = self._evaluate_chunk(0, 3)
            results_end = self._evaluate_chunk(num_items-3, 3)
            return '[\n' + ',\n'.join([repr(r) for r in results_beg]) + \
                ',\n...\n' + ',\n'.join([repr(r) for r in results_end]) + '\n]'
        else:
            # evaluate all items in expression
            results = self._evaluate_chunk(0, num_items)
            return '[\n' + ',\n'.join([repr(r) for r in results])+'\n]'

    def __iter__(self):
        # todo: check efficiency
        num_items = len(self)
        chunk_size = max(_min_chunk_size,
                         min(_max_chunk_size,
                             int(math.ceil(float(num_items) / _num_chunks))))
        for i in range(0, num_items, chunk_size):
            results = self._evaluate_chunk(i, chunk_size)
            for r in results:
                yield r

    def __getitem__(self, key):
        # todo: hard to understand, needs some comments
        if isinstance(key, expression):
            return index_op(self, key, slice(None, None, None))
        elif isinstance(key, tuple) and len(key) == 2 and (
                isinstance(key[0], expression) and isinstance(key[1], slice) or
                isinstance(key[0], slice) and isinstance(key[1], expression) or
                isinstance(key[0], slice) and isinstance(key[1], slice)):
            return index_op(self, key[0], key[1])
        elif isinstance(key, int):
            if key >= len(self):
                raise IndexError('list index out of range')
            return self._evaluate_chunk(key, 1)[0]
        else:
            raise TypeError('unrecognized index key type %s' % type(key))

    def __setitem__(self, key):
        raise RuntimeError('not implemented')

    def __getattribute__(self, name):
        if name == 'T':
            return TRANSPOSE(self)
        else:
            return super(expression, self).__getattribute__(name)

    def _evaluate_chunk(self, index, size, use_cache=False):
        if type(self) == 'expression':
            raise RuntimeError('not meant to be called on expression')
        if not use_cache:
            return self._evaluate_chunk(index, size, use_cache)
        if hasattr(self, '_cache_index') and \
                self._cache_index == (index, size):
            return self._cache_value
        else:
            self._cache_index = (index, size)
            self._cache_value = self._evaluate_chunk(index, size, use_cache)
            return self._cache_value

    # call this method to convert expression to list of results
    def evaluate(self, num_chunks=_num_chunks, use_cache=True):
        num_items = len(self)
        chunk_size = max(_min_chunk_size,
                         min(_max_chunk_size,
                             int(math.ceil(float(num_items) / num_chunks))))
        results = []
        for i in range(0, num_items, chunk_size):
            s = min(num_items, i + chunk_size) - i
            results.extend(self._evaluate_chunk(i, s, use_cache))
        return results

    # call this method to split a list of tuples into a tuple of list
    def unzip(self):
        if len(self) == 0:
            return ()
        first_item = self._evaluate_chunk(0, 1)[0]
        if not isinstance(first_item, tuple):
            raise TypeError('Can only unzip list of tuples')
        tuple_size = len(first_item)
        results = ()
        for i in xrange(tuple_size):
            results += (select_op(self, i), )
        return results


class unary_op(expression):
    def __init__(self, operand, op, kwargs={}):
        operand = _make_expression(operand)
        self.operand = operand
        self.op = op
        self.kwargs = kwargs

    def __len__(self):
        return len(self.operand)

    def __repr__(self):
        return 'Expression %s %s (%d elements)' % \
            (self.op, type(self), len(self)) + expression.__repr__(self)

    @staticmethod
    def _check_operands(operand):
        pass

    def _evaluate_chunk(self, index, size, use_cache=False):
        operandlist = expression._evaluate_chunk(self.operand,
                                                 index, size, use_cache)
        return self.op(operandlist, **self.kwargs)


class binary_op(expression):
    def __init__(self, left, right, op):
        self.left = _make_expression(left)
        self.right = _make_expression(right)
        binary_op._check_operands(self.left, self.right)
        self.op = op

    def __len__(self):
        return max(len(self.left), len(self.right))

    def __repr__(self):
        return 'Expression %s %s (%d elements)' % \
            (self.op, type(self), len(self)) + expression.__repr__(self)

    @staticmethod
    def _check_operands(left, right):
        # Assumes both left and right are instances of expressions
        # At least one of the following must be true
        #    1. both operands has same length
        #    2. left has length 1
        #    3. right has length 1
        if len(left) != len(right) and len(left) != 1 and len(right) != 1:
            raise ValueError('attempting to operate on incompatible lengths' +
                             '%d and %d' % (len(left), len(right)))

    def _evaluate_chunk(self, index, size, use_cache=False):
        leftlist = []
        if len(self.left) == 1:
            leftlist = expression._evaluate_chunk(self.left,
                                                  0, 1, use_cache)
        else:
            leftlist = expression._evaluate_chunk(self.left,
                                                  index, size, use_cache)
        rightlist = []
        if len(self.right) == 1:
            rightlist = expression._evaluate_chunk(self.right,
                                                   0, 1, use_cache)
        else:
            rightlist = expression._evaluate_chunk(self.right,
                                                   index, size, use_cache)
        return self.op(leftlist, rightlist)


class ternary_op(expression):
    pass


class list_expression(expression):
    def __init__(self, x):
        if isinstance(x, list):
            self.items = copy.copy(x)
            # note: prefer copy.copy(x) over just storing reference to x.
            # this way self.items is unaffected by modifications
            # (append, insert, etc) to a separate reference of x
        else:
            self.items = [x]

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return 'Expression %s (%d elements)' % (type(self), len(self)) \
            + expression.__repr__(self)

    @staticmethod
    def _check_operands():
        pass

    def _evaluate_chunk(self, index, size, use_cache=False):
        return self.items[index:index + size]


class index_op(expression):
    def __init__(self, src, row_sel, col_sel):
        self.res_len = index_op._check_operands(src, row_sel, col_sel)
        self.src = src
        self.row_sel = row_sel
        self.col_sel = col_sel

    def __len__(self):
        return self.res_len

    def __repr__(self):
        return 'Expression %s (%d elements)' % (type(self), len(self)) \
            + expression.__repr__(self)

    @staticmethod
    def _check_operands(src, row_sel, col_sel):
        if not isinstance(src, expression):
            TypeError('Invalid source operand type %s' % type(src))
        sel_len = None
        if isinstance(row_sel, expression) and isinstance(col_sel, slice):
            sel_len = len(row_sel)
        elif isinstance(row_sel, slice) and isinstance(col_sel, expression):
            sel_len = len(col_sel)
        elif isinstance(row_sel, slice) and isinstance(col_sel, slice):
            sel_len = 1
        else:
            raise TypeError('Incompatible row + column selector types' +
                            '(%s, %s)' % (type(row_sel), type(col_sel)))
        if len(src) != sel_len and len(src) != 1 and sel_len != 1:
            raise ValueError('Incompatible source and selector lengths' +
                             '(%d and %d)' % (len(src), sel_len))
        return max(sel_len, len(src))

    def _evaluate_chunk(self, index, size, use_cache=False):
        srclist = []
        if len(self.src) == 1:
            srclist = expression._evaluate_chunk(self.src,
                                                 0, 1, use_cache)
        else:
            srclist = expression._evaluate_chunk(self.src,
                                                 index, size, use_cache)
        selector = [None, None]
        if isinstance(self.row_sel, slice):
            selector[0] = self.row_sel
        elif len(self.row_sel) == 1:
            selector[0] = expression._evaluate_chunk(self.row_sel,
                                                     0, 1, use_cache)
        else:
            selector[0] = expression._evaluate_chunk(self.row_sel,
                                                     index, size, use_cache)
        if isinstance(self.col_sel, slice):
            selector[1] = self.col_sel
        elif len(self.col_sel) == 1:
            selector[1] = expression._evaluate_chunk(self.col_sel,
                                                     0, 1, use_cache)
        else:
            selector[1] = expression._evaluate_chunk(self.col_sel,
                                                     index, size, use_cache)
        return vfuncs._idx(srclist, tuple(selector))


class select_op(expression):
    def __init__(self, operand, index):
        select_op._check_operands(operand, index)
        self.operand = operand
        self.index = index

    def __len__(self):
        return len(self.operand)

    def __repr__(self):
        return 'Expression %s (%d elements)' % (type(self), len(self)) \
            + expression.__repr__(self)

    @staticmethod
    def _check_operands(operand, index):
        if not isinstance(operand, expression):
            raise TypeError('Invalid operand type %s' % type(opernad))
        if not isinstance(index, (int, long)):
            raise TypeError('Invalid index type %s' % type(index))

    def _evaluate_chunk(self, index, size, use_cache=False):
        operandlist = expression._evaluate_chunk(self.operand,
                                                 index, size, use_cache)
        return [x[self.index] for x in operandlist]


def _make_expression(x):
    if isinstance(x, expression):
        return x
    else:
        return list_expression(x)


class nbhds_op(expression):
    def __init__(self, data, queries, k, r):
        self.data = data
        self.queries = queries
        self.k = k
        self.r = r

    def __len__(self):
        if self.queries is None:
            return self.data.shape[0]
        else:
            return len(self.queries)

    def __repr__(self):
        # todo: should this be defined in parent class instead?
        return 'Expression %s (%d elements)' % (type(self), len(self)) \
            + expression.__repr__(self)

    def _evaluate_chunk(self, index, size, use_cache=False):
        self.data._update_kd_tree()
        if self.queries is None:
            indices = np.arange(index, index + size)
            nhbrs = kdtree._query(self.data._tree, indices, self.k, self.r)
        else:
            nhbrs = kdtree._query(self.data._tree,
                                  self.queries[index:index + size],
                                  self.k, self.r)
        return nhbrs
