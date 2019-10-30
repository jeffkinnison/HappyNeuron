#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from setuptools import setup, find_packages


def collect_entry_points():
    entry_points = []
    base = os.path.abspath(os.path.dirname(__file__))
    for root, dirs, files in os.walk(os.path.join(base, 'happyneuron')):
        for fname in files:
            if os.path.splitext(fname)[1] == '.py' and not re.search(r'^__.+__[.]py$', fname):
                fpath = os.path.join(root, fname)
                with open(fpath, 'r') as f:
                    for line in f.readlines():
                        if re.search(r'^def main[(][)]:', line):
                            cmdname = os.path.splitext(fname)[0]
                            modulepath = os.path.splitext(fpath.replace(base, '').strip('/').replace('/', '.'))[0]
                            entry_points.append('{} = {}:main'.format(cmdname, modulepath))
    return entry_points


setup(
    name='HappyNeuron',
    packages=find_packages(exclude=['test*', 'docs*']),
    version=open('VERSION').read().strip(),
    include_package_data=True,
    zip_safe=False,
    author='Rafael Vescovi, Hanyu Li, Jeff Kinnison, Nicola Ferrier, Thomas Uram, Misha Salin, Murat Keceli',
    author_email='ravescovi@anl.gov',
    description='Exascale pipeline for processing neural microscopy data.',
    keywords=['neuroscience',
              'microscopy',
              'imaging',
              'alignment',
              'segmentation',
              'computer vision',
              'deep learning'],
    download_url='http://github.com/ravescovi/HappyNeuron',
    license='BSD-3',
    platforms='Any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'],
    entry_points={
        'console_scripts': collect_entry_points()
    }
)
