from setuptools import setup, find_packages

setup(name='sparse_to_dense',
      version='1.0',
      packages=['sparse_to_dense'],
      install_requires=[
          'matplotlib',
          'h5py',
          'imageio',
          'opencv-python',
      ],
      )
