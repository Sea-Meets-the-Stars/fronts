
# Standard imports
import glob, os
from setuptools import setup, find_packages


# Begin setup
setup_keywords = dict()
setup_keywords['name'] = 'fronts'
setup_keywords['description'] = 'Fronts in the Ocean'
setup_keywords['author'] = 'J. Xavier Prochaska, P. Cornillon'
setup_keywords['author_email'] = 'jxp@ucsc.edu'
setup_keywords['license'] = 'BSD'
setup_keywords['url'] = 'https://github.com/AI-for-Ocean-Science/fronts'
setup_keywords['version'] = '0.0.dev0'
# Use README.rst as long_description.
setup_keywords['long_description'] = ''
if os.path.exists('README.md'):
    with open('README.md') as readme:
        setup_keywords['long_description'] = readme.read()
setup_keywords['provides'] = [setup_keywords['name']]
setup_keywords['requires'] = ['Python (>=3.9.0)']
setup_keywords['install_requires'] = [
    'torch',  
    'torchvision', 'seaborn', 'smart-open[s3]', 'pyarrow',
    'scikit-learn', 'scikit-image', 'tqdm', 'astropy', 'astropy-healpix',
    'healpy', 'cftime', 'bokeh', 'umap-learn', 'llvmlite', 'boto3',
    'xarray', 'h5netcdf', 'emcee', 'corner',
    'importlib-metadata', 'timm==0.3.2', 'xmitgcm', 'dask', 's3fs',
    'ujson', 'zarr',
    'scikit-learn', 'scikit-image', 'tqdm', 'gsw']
setup_keywords['zip_safe'] = False
setup_keywords['use_2to3'] = False
setup_keywords['packages'] = find_packages()
setup_keywords['setup_requires'] = ['pytest-runner']
setup_keywords['tests_require'] = ['pytest']

if os.path.isdir('bin'):
    setup_keywords['scripts'] = [fname for fname in glob.glob(os.path.join('bin', '*'))
                                 if not os.path.basename(fname).endswith('.rst')]

setup(**setup_keywords)
