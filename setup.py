from distutils.core import setup

setup(
    name='PyNFG',
    version='0.1.0',
    author='James Bono',
    author_email='jwbono@gmail.com',
    packages=['pynfg', 'pynfg.test'],
    scripts=['bin/hideandseek.py'],
    url='http://pypi.python.org/pypi/PyNFG/',
    license='LICENSE.txt',
    description='A Python package for implementing and solving Network form games.',
    long_description=open('README.txt').read(),
    install_requires=[
        "matplotlib",
        "networkx",
        "scipy",
        "numpy",
        
    ],
)