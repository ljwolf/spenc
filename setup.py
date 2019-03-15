from setuptools import setup
import os

basepath = os.path.dirname(__file__)
init = os.path.join(basepath, 'spenc/__init__.py')

with open(init, 'r') as initfile:
    firstline = initfile.readline()
init_version = firstline.split('=')[-1].strip()

setup(name='spenc',
      version=init_version,
      description='Spatially-Encouraged Spectral Clustering, a method of discovering clusters/deriving labels for spatially-referenced data with attribute/labels attached',
      url='https://github.com/ljwolf/spenc',
      author='Levi John Wolf',
      author_email='levi.john.wolf@gmail.com',
      license='3-Clause BSD',
      python_requires='>=3.5',
      packages=['spenc'],
      install_requires=['scikit-learn', 'scipy'])
