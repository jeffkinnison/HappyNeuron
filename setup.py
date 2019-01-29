#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='autoem',
    packages=find_packages(exclude=['test*']),
    version=open('VERSION').read().strip(),
    include_package_data=True,
    zip_safe=False,
    author='Rafael Vescovi, Hanyu Li, Nicola Ferrier, Thomas Uram, Wushi Dong, Murat Keceli',
    author_email='ravescovi@anl.gov',
    description='AutoEM',
    keywords=['Electron Microscopy', 
              'brain',
              'flood fill netword',
              'imaging'],
    download_url='http://github.com/ravescovi/autoem',
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
        'Programming Language :: Python :: 3.5']
)
