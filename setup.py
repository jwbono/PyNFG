from distutils.core import setup
from setuptools import find_packages

setup(
    name='PyNFG',
    version='0.1.2',
    author='James Bono',
    author_email='jwbono@gmail.com',
    packages= find_packages(),
    scripts=['bin/hideandseek.py', 'bin/stackelberg.py'],
    url='http://pypi.python.org/pypi/PyNFG/',
    license='LICENSE.txt',
    description='A Python package for implementing and solving Network form games.',
    long_description=open('README.txt').read(),
    install_requires=[
        "matplotlib >= 1.1.1",
        "networkx >= 1.7",
        "scipy >= 0.7.0",
        "numpy >= 1.7",
        "pygraphviz >= 1.1",
        ],
    classifiers = [
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        ],
)