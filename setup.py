from setuptools import setup

setup(name='spenc',
      version='0.1',
      description='Spatially-Encouraged Spectral Clustering, a method of discovering clusters/deriving labels for spatially-referenced data with attribute/labels attached',
      url='https://github.com/ljwolf/spenc',
      author='Levi John Wolf',
      author_email='levi.john.wolf@gmail.com',
      license='3-Clause BSD',
      python_requires='>=3.5',
      packages=['spenc'],
      install_requires=['scikit-learn', 'scipy'])
