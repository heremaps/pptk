import os
import os.path
import sys
import platform


def _add_path():
    _root_dir = os.path.dirname(os.path.realpath(__file__))

    # manual specify list of dll directories, with paths relative to _root_dir
    _dll_dirs = ['libs']

    if platform.system() == 'Windows':
        os.environ.setdefault('PATH', '')
        paths = os.environ['PATH'].split(';')
        for x in _dll_dirs:
            x = os.path.join(_root_dir, x)
            if os.path.isdir(x) and x not in paths:
                paths = [x] + paths
        os.environ['PATH'] = ';'.join(paths)


_add_path()
