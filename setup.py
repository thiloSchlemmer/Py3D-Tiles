# -*- coding: utf-8 -*-

import os
# HACK for `nose.collector` to work on python 2.7.3 and earlier
import multiprocessing
from setuptools import setup, find_packages

# HACK READTHEDOCS (find a better solution)
if '/home/docs/checkouts/readthedocs' in os.getcwd():
    requires = []
else:
    requires = ['shapely', 'numpy']

setup(name='Py3d_Tiles',
      version='0.1',
      description='Cesium 3D-Tiles format reader and writer',
      author=u'Thilo Schlemmer',
      author_email='thio.schlemmer@gmail.com',
      license='MIT',
      url='https://github.com/thiloSchlemmer/Py3d-Tiles',
      packages=find_packages(exclude=['tests', 'doc']),
      zip_safe=False,
      test_suite='nose.collector',
      install_requires=requires,
      )
