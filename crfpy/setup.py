from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('alpha_beta', ['alpha_beta.pyx'], include_dirs = [np.get_include()],library_dirs=["/home/rjadams/lib"]),
    ]

setup(
    ext_modules = cythonize(extensions)
    )