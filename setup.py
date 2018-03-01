from setuptools import setup, find_packages
from distutils.extension import Extension
import subprocess
import os
import os.path
import shutil
import platform

from pip import wheel

wheel_tags = wheel.pep425tags.get_supported()[0]

system_type = platform.system()


def make_mod(x):
    if system_type == 'Windows':
        return x + '.pyd'
    elif system_type == 'Linux':
        return x + '.so'
    elif system_type == 'Darwin':
        return x + '.so'
    else:
        raise RuntimeError('Unknown system type %s', system_type)


def make_lib(x, version_suffix=''):
    if system_type == 'Windows':
        return x + '.dll'
    elif system_type == 'Linux':
        return 'lib' + x + '.so' + version_suffix
    elif system_type == 'Darwin':
        return 'lib' + x + '.dylib'
    else:
        raise RuntimeError('Unknown system type %s', system_type)


def make_exe(x):
    if system_type == 'Windows':
        return x + '.exe'
    else:
        return x


def list_libs():
    libs_dir = os.path.join('pypcl', 'libs')
    exclude_list = ['Makefile', 'cmake_install.cmake']
    return [f for f in os.listdir(libs_dir)
            if os.path.isfile(os.path.join(libs_dir, f))
            and f not in exclude_list]


setup(
    name='pypcl',
    version='0.1.0',
    description='A Python package for facilitating point cloud processing.',
    author='Victor Lu',
    packages=find_packages(),
    package_data={
        'pypcl': [
            os.path.join('libs', f) for f in list_libs()] + [
            os.path.join('libs',
                         'qt_plugins', 'platforms', make_lib('*', '*')),
            os.path.join('libs',
                         'qt_plugins', 'xcbglintegrations', make_lib('*', '*'))
            ],
        'pypcl.kdtree': [make_mod('kdtree')],
        'pypcl.processing.estimate_normals': [make_mod('estimate_normals')],
        'pypcl.vfuncs': [make_mod('vfuncs')],
        'pypcl.viewer': [make_exe('viewer'), 'qt.conf']},
    options={'bdist_wheel': {
        'python_tag': wheel_tags[0],
        'plat_name': wheel_tags[2]}})
