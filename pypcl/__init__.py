import os
import os.path
import sys
import platform

_root_dir = os.path.dirname(os.path.realpath(__file__))

# manual specify list of dll directories, with paths relative to _root_dir
_dll_dirs = ['libs']

if platform.system() == 'Windows':
    os.environ.setdefault('PATH','')
    paths = os.environ['PATH'].split(';')
    for x in _dll_dirs:
        x = os.path.join(_root_dir, x)
        if os.path.isdir(x) and x not in paths:
            paths = [x] + paths
    os.environ['PATH'] = ';'.join(paths)

from viewer import *
from points import *
from kdtree import kdtree
from processing.estimate_normals.estimate_normals import estimate_normals

# need a more scalable way for handling additions of new functions
__all__ = [
    'viewer',
    'points',
    'expr',
    'kdtree',
    'estimate_normals',
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
    'TRANSPOSE']