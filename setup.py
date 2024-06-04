import os
from setuptools import setup, find_packages


if os.path.isfile('VERSION'):
  with open('VERSION') as f:
    VERSION = f.read()
else:
  VERSION = '0.0.dev0'

with open('README.md') as f:
  README = f.read()

setup(name='sublora',
      description='SubLoRA Compression Bounds for LLMs',
      long_description=README,
      long_description_content_type='text/markdown',
      version=VERSION,
      license='Apache License 2.0',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      packages=find_packages(exclude=[
          'config',
          'config.*',
          'data',
          'data.*',
          'experiments',
          'experiments.*',
      ]),
      extras_require={}
     )