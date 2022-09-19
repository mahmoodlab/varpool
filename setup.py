from setuptools import setup, find_packages
import os

version = None
with open(os.path.join('var_pool', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


install_requires = []  # ['pycox']


setup(name='var_pool',
      version=version,
      description='Reproduce experiments for variance pooling MICCAI paper.',
      author='Iain Carmichael',
      author_email='icarmichael@bwh.harvard.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
